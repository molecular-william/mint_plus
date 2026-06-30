#!/usr/bin/env python3
"""
Memory-efficient, parallelized, Numba-accelerated STRING preparation engine.
Optimized to handle billions of links without intermediate memory bloat or slow lookups.
"""

import gzip
import zstandard as zstd
import random
import sys
import gc
from pathlib import Path
import numpy as np
import numba as nb
from tqdm import tqdm
from mint_plus.utils.log import get_logger

logger = get_logger(__name__)

# =====================================================================
# NUMBA ACCELERATED COMPUTATIONAL KERNELS
# =====================================================================

@nb.njit(parallel=True, cache=True)
def generate_pair_keys_numba(p1, p2, prot_id_to_cluster_id):
    """
    Fuses cluster mapping, min/max calculations, and 64-bit key packing 
    into a single parallelized pass over RAM. No intermediate arrays are created.
    """
    n = len(p1)
    pair_keys = np.empty(n, dtype=np.uint64)
    
    for i in nb.prange(n):
        c1 = prot_id_to_cluster_id[p1[i]]
        c2 = prot_id_to_cluster_id[p2[i]]
        
        if c1 < c2:
            min_c = np.uint64(c1)
            max_c = np.uint64(c2)
        else:
            min_c = np.uint64(c2)
            max_c = np.uint64(c1)
            
        pair_keys[i] = (min_c << 32) | max_c
        
    return pair_keys


@nb.njit(parallel=True, cache=True)
def filter_leaking_train_numba(p1_ids, p2_ids, train_file_indices, val_clus_mask, prot_id_to_cluster_id):
    """
    Parallel check determining which training rows leak validation clusters.
    Returns the filtered indices directly.
    """
    n = len(train_file_indices)
    keep_mask = np.empty(n, dtype=nb.bool_)
    
    for i in nb.prange(n):
        idx = train_file_indices[i]
        c1 = prot_id_to_cluster_id[p1_ids[idx]]
        c2 = prot_id_to_cluster_id[p2_ids[idx]]
        
        # Keep only if neither cluster is marked true in the validation mask
        keep_mask[i] = not (val_clus_mask[c1] or val_clus_mask[c2])
        
    return train_file_indices[keep_mask]


# =====================================================================
# MAIN PIPELINE ENGINE
# =====================================================================

def prepare_data_stream_optimized(
    sequences_file: str,
    clusters_file: str,
    links_file: str,
    output_dir: str = ".",
    val_size: int = 250000
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Map Clusters
    logger.info("=== Reading Cluster Representatives ====")
    reps = {}
    with open(clusters_file, "r") as f:
        for line in tqdm(f):
            parts = line.strip().split()
            rep, seq = parts[0], parts[1]
            reps[sys.intern(seq)] = sys.intern(rep)

    unique_prots = list(reps.keys())
    prot_to_id = {name: idx for idx, name in enumerate(unique_prots)}
    
    unique_clusters = list(set(reps.values()))
    cluster_to_id = {name: idx for idx, name in enumerate(unique_clusters)}
    
    # Downcast to int32 because total unique clusters easily fit in 2 billion limits
    prot_id_to_cluster_id = np.array([cluster_to_id[reps[p]] for p in unique_prots], dtype=np.int32)
    
    del unique_prots, unique_clusters, cluster_to_id
    gc.collect()

    total_links = 2_365_172_266  # magic number for STRINGDB v12
    
    # 4. Replicate Shuffle #1 via Integer Layouts
    logger.info("=== Simulating Shuffle 1 ====")
    line_order = np.arange(total_links, dtype=np.int64)
    
    # Use modern Generator for up to 2-3x quicker shuffling over legacy np.random.seed
    rng = np.random.default_rng(137)
    rng.shuffle(line_order)

    # Memory downcasting optimization: int32 handles up to 2.1 billion unique names
    p1_ids = np.empty(total_links, dtype=np.int32)
    p2_ids = np.empty(total_links, dtype=np.int32)
    
    logger.info("=== Mapping link tokens to memory-efficient IDs ====")
    with gzip.open(links_file, "rt") as f:
        next(f)  # Skip header
        for idx, line in tqdm(enumerate(f), total=total_links, desc="Mapping tokens"):
            parts = line.strip().split()
            p1_ids[idx] = prot_to_id[parts[0]]
            p2_ids[idx] = prot_to_id[parts[1]]

    # Shuffling layout arrays
    p1_shuffled1 = p1_ids[line_order]
    p2_shuffled1 = p2_ids[line_order]
    gc.collect()
    
    # 5. Precise Cluster Deduplication using Numba
    logger.info("=== Filtering Unique Cluster Links (Numba Parallel) ====")
    pair_keys = generate_pair_keys_numba(p1_shuffled1, p2_shuffled1, prot_id_to_cluster_id)
    
    del p1_shuffled1, p2_shuffled1
    gc.collect()

    # Getting and sorting unique indices
    _, unique_indices = np.unique(pair_keys, return_index=True)
    del pair_keys
    unique_indices.sort()

    surviving_file_indices = line_order[unique_indices]
    del unique_indices, line_order
    gc.collect()
    logger.info(f"Kept {len(surviving_file_indices)} / {total_links} links.")

    # 6. Replicate Shuffle #2
    logger.info("=== Simulating Shuffle 2 ====")
    rng_split = np.random.default_rng(731)
    rng_split.shuffle(surviving_file_indices)

    val_file_indices = surviving_file_indices[:val_size]
    train_file_indices = surviving_file_indices[val_size:]
    del surviving_file_indices

    # Identify leaking validation clusters perfectly
    val_clus_mask = np.zeros(len(prot_id_to_cluster_id), dtype=np.bool_)
    val_clus_mask[prot_id_to_cluster_id[p1_ids[val_file_indices]]] = True
    val_clus_mask[prot_id_to_cluster_id[p2_ids[val_file_indices]]] = True

    # Build filtered training indexes via parallel Numba function
    filtered_train_file_indices = filter_leaking_train_numba(
        p1_ids, p2_ids, train_file_indices, val_clus_mask, prot_id_to_cluster_id
    )
    
    del train_file_indices, val_clus_mask, prot_id_to_cluster_id
    gc.collect()

    # Create O(1) boolean masks for streaming lookup instead of slow binary searches
    is_val = np.zeros(total_links, dtype=bool)
    is_val[val_file_indices] = True
    del val_file_indices

    is_train_filtered = np.zeros(total_links, dtype=bool)
    is_train_filtered[filtered_train_file_indices] = True
    del filtered_train_file_indices
    gc.collect()

    # 7. Pass 2: Stream Outputs
    logger.info("=== High-Speed FASTA Indexing ====")
    seqs = {}
    with open(sequences_file, "r") as f:
        current_name = None
        chunks = []
        for line in tqdm(f):
            if line.startswith(">"):
                if current_name:
                    seqs[sys.intern(current_name)] = "".join(chunks)
                current_name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if current_name:
            seqs[sys.intern(current_name)] = "".join(chunks)

    logger.info("=== Final Pass: Streaming exact matching files to disk ====")
    written_val_seqs = set()
    written_filtered_seqs = set()

    num_train_link = 0
    with gzip.open(links_file, "rt") as f_src, \
         zstd.open(output_dir / "validation.links.txt.zst", "wt", encoding="ascii") as f_v_lnk, \
         zstd.open(output_dir / "validation.seqs.txt.zst", "wt", encoding="ascii") as f_v_seq, \
         zstd.open(output_dir / "training_filtered.links.txt.zst", "wt", encoding="ascii") as f_tf_lnk, \
         zstd.open(output_dir / "training_filtered.seqs.txt.zst", "wt", encoding="ascii") as f_tf_seq:

        next(f_src)  # Skip Header
        
        for idx, line in tqdm(enumerate(f_src), total=total_links, desc="Streaming outputs"):
            val_target = is_val[idx]
            train_target = is_train_filtered[idx]
            
            # Instant skip for unneeded rows
            if not (val_target or train_target):
                continue
                
            line_str = line.strip()
            parts = line_str.split()
            name1, name2 = parts[0], parts[1]
            
            if val_target:
                f_v_lnk.write(line_str + "\n")
                if name1 not in written_val_seqs:
                    f_v_seq.write(f"{name1} {seqs[name1]}\n")
                    written_val_seqs.add(name1)
                if name2 not in written_val_seqs:
                    f_v_seq.write(f"{name2} {seqs[name2]}\n")
                    written_val_seqs.add(name2)
            elif train_target:
                f_tf_lnk.write(line_str + "\n")
                if name1 not in written_filtered_seqs:
                    f_tf_seq.write(f"{name1} {seqs[name1]}\n")
                    written_filtered_seqs.add(name1)
                if name2 not in written_filtered_seqs:
                    f_tf_seq.write(f"{name2} {seqs[name2]}\n")
                    written_filtered_seqs.add(name2)
                num_train_link += 1

    logger.info(f"Finished! {num_train_link} training links!")