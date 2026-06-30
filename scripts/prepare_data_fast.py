#!/usr/bin/env python3
"""
Memory-efficient, parallelized, Numba-accelerated STRING preparation engine.

Optimizations over the baseline prepare_data.py:
  1. Parallel gzip decompression via rapidgzip / bgzip
  2. Parallel zstd compression via python-zstandard
  3. Optional GPU-accelerated uint64 sort via cupy
  4. Single-mmap architecture (decompress once, process twice)
  5. On-disk cache for prot_to_id, prot_id_to_cluster_id, surviving_file_indices
  6. Multi-process first pass (map reduce over mmap segments)
  7. Pre-extracted sequences (single sequential FASTA scan, no random seeks)
  8. FASTA index cached for reuse

Usage:
    # Default — uses best available tools, falls back gracefully
    python scripts/prepare_data_fast.py \
        --sequences ./data/protein.sequences.v12.0.fasta \
        --clusters ./data/clusters.v12.0.txt \
        --links ./data/protein.links.v12.0.txt.gz \
        --output_dir ./data/diamond

    # Force CPU sort (skip GPU even if cupy is available)
    python scripts/prepare_data_fast.py ... --cpu-sort

    # Override decompression threads on a fat node
    python scripts/prepare_data_fast.py ... --decompress-threads 32

    # Bypass cache and recompute everything
    python scripts/prepare_data_fast.py ... --no-cache

    # Custom cache directory (defaults to output_dir)
    python scripts/prepare_data_fast.py ... --cache-dir /fast/scratch/cache
"""

# Use python-isal (Intel ISA-L) for 2-3x faster gzip fallback when available,
# otherwise fall back to Python's built-in gzip.
try:
    import isal.igzip as _gzip_backend
    _HAS_ISAL = True
except ImportError:
    import gzip as _gzip_backend
    _HAS_ISAL = False
import json
import mmap
import multiprocessing as mp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'

import pickle
import subprocess
import shutil
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import numba as nb
from tqdm import tqdm

# Add project root to python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mint_plus.utils.log import get_logger

logger = get_logger(__name__)

# =====================================================================
# PARALLEL GZIP I/O HELPERS
# =====================================================================

def _find_tool(name):
    """Find a system tool path, return None if not found."""
    return shutil.which(name)


def _open_zst_write(path, num_threads=16):
    """
    Open a zstd file for multithreaded writing via python-zstandard.
    Returns (stdin_pipe, cleanup_fn) tuple.
    
    Write to the pipe as if it were a normal file object, then call
    cleanup_fn() to finalize the zstd file.
    """
    import io
    import zstandard as zstd
    
    cctx = zstd.ZstdCompressor(level=3, threads=num_threads)
    path = str(path)
    f_out = open(path, "wb")
    zstd_writer = cctx.stream_writer(f_out)
    text_wrapper = io.TextIOWrapper(zstd_writer, encoding='ascii',
                                    write_through=True)
    
    def _cleanup():
        text_wrapper.flush()
        try:
            underlying = text_wrapper.detach()
        except ValueError:
            pass
        else:
            underlying.close()
    
    return text_wrapper, _cleanup


# =====================================================================
# NUMBA ACCELERATED COMPUTATIONAL KERNELS
# =====================================================================

@nb.njit(parallel=True, cache=True)
def generate_pair_keys_numba(p1, p2, prot_id_to_cluster_id):
    """
    Fuses cluster mapping, min/max pair ordering, and 64-bit key packing
    into a single parallelized pass over RAM. No intermediate arrays.
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
def filter_leaking_train_numba(p1_ids, p2_ids, train_file_indices,
                                val_clus_mask, prot_id_to_cluster_id):
    """
    Parallel check: which training rows leak validation clusters.
    Returns filtered indices directly.
    """
    n = len(train_file_indices)
    keep_mask = np.empty(n, dtype=nb.bool_)
    for i in nb.prange(n):
        idx = train_file_indices[i]
        c1 = prot_id_to_cluster_id[p1_ids[idx]]
        c2 = prot_id_to_cluster_id[p2_ids[idx]]
        keep_mask[i] = not (val_clus_mask[c1] or val_clus_mask[c2])
    return train_file_indices[keep_mask]


# =====================================================================
# NUMBA-FRIENDLY PROTEIN NAME HASH TABLE
# =====================================================================
# Replaces Python dict lookup + .decode('ascii') with a hash table
# operating directly on raw bytes inside Numba. This avoids allocating
# 4.6 billion Python strings in the first pass.
#
# Data structures (all flat NumPy arrays, Numba-compatible):
#   name_buf       — uint8[N_bytes], all names concatenated
#   name_offsets   — int64[N], byte offset of each name in name_buf
#   name_lengths   — int32[N], byte length of each name
#   name_hashes    — uint32[N], DJB2 hash of each name
#   ids            — int32[N], the integer ID for each name
#   hash_table     — int32[T], slots = next_pow2(2*N), -1 = empty
#   chain_next     — int32[N], collision chain link (-1 = end)

@nb.njit
def _djb2_hash(buf, start, length):
    """Compute DJB2 hash of buf[start:start+length] (32-bit, wrapping)."""
    h = 5381
    for i in range(length):
        h = (h * 33 + buf[start + i]) & 0xFFFFFFFF
    return np.uint32(h)


@nb.njit
def _bytes_equal(buf_a, start_a, buf_b, start_b, length):
    """Return True if the two byte sequences are identical."""
    for i in range(length):
        if buf_a[start_a + i] != buf_b[start_b + i]:
            return False
    return True


@nb.njit
def _lookup_name(buf_bytes, start, length,
                 hash_table, chain_next,
                 name_buf, name_offsets, name_lengths,
                 name_hashes, ids):
    """
    Look up a protein name from raw bytes.
    
    Returns the integer ID, or -1 if not found.
    All arrays are flat NumPy arrays (no Python objects).
    """
    h = _djb2_hash(buf_bytes, start, length)
    table_mask = len(hash_table) - 1
    slot = h & table_mask
    idx = hash_table[slot]

    while idx != -1:
        if name_hashes[idx] == h and name_lengths[idx] == length:
            off = name_offsets[idx]
            if _bytes_equal(buf_bytes, start, name_buf, off, length):
                return ids[idx]
        idx = chain_next[idx]
    return -1


def _build_numba_hash_table(prot_to_id):
    """
    Build Numba-friendly hash table from a Python dict.
    
    Args:
        prot_to_id: dict[str, int] — protein name → integer ID
    
    Returns:
        tuple of 7 arrays:
        (name_buf, name_offsets, name_lengths, name_hashes,
         ids, hash_table, chain_next)
    """
    num_prots = len(prot_to_id)

    # Encode all names to bytes and compute hashes
    name_items = []
    for name, pid in prot_to_id.items():
        name_bytes = name.encode('ascii')
        # Pre-compute hash using numpy view for consistency with Numba
        nb_view = np.frombuffer(name_bytes, dtype=np.uint8)
        h = _djb2_hash(nb_view, 0, len(name_bytes))
        name_items.append((name_bytes, h, pid))

    # Sort for deterministic ordering of chain entries
    name_items.sort(key=lambda x: x[0])

    # Concatenate all names into a single buffer
    name_parts = [item[0] for item in name_items]
    name_buf = np.frombuffer(b''.join(name_parts), dtype=np.uint8)

    # Flat arrays
    name_offsets = np.empty(num_prots, dtype=np.int64)
    name_lengths = np.empty(num_prots, dtype=np.int32)
    name_hashes = np.empty(num_prots, dtype=np.uint32)
    ids = np.empty(num_prots, dtype=np.int32)
    chain_next = np.empty(num_prots, dtype=np.int32)

    # Hash table: next power of 2 >= 2 * num_prots
    table_size = 1
    while table_size < num_prots * 2:
        table_size <<= 1
    hash_table = np.full(table_size, -1, dtype=np.int32)

    buf_offset = 0
    for i, (name_bytes, h, pid) in enumerate(name_items):
        name_offsets[i] = buf_offset
        name_lengths[i] = len(name_bytes)
        name_hashes[i] = h
        ids[i] = pid
        buf_offset += len(name_bytes)

        slot = h & (table_size - 1)
        chain_next[i] = hash_table[slot]
        hash_table[slot] = i

    return (name_buf, name_offsets, name_lengths, name_hashes,
            ids, hash_table, chain_next)


# Module-level global shared with worker processes via fork COW.
# Set in prepare_data_fast() after building the hash table,
# used by _worker_process_chunk().
_WORKER_HASH_DATA = None


# =====================================================================
# MULTIPROCESS WORKER — FIRST PASS MAP REDUCE
# =====================================================================

def _worker_process_chunk(args):
    """
    Map worker for the first-pass link-to-ID mapping.
    
    Each worker:
      1. Opens the decompressed temp file and mmap's its byte segment
      2. Parses each line using Numba hash lookup on raw bytes
         (no Python string decode, no dict)
      3. Writes results into pre-allocated memmap arrays at a known offset
      4. Returns the number of lines actually processed
    
    The prot_to_id hash table arrays are inherited via fork COW through
    the module-level _WORKER_HASH_DATA global — no pickle overhead.
    
    Args via tuple:
        tmp_path, start, end, p1_mmap_path, p2_mmap_path, offset
    
    Returns: int (number of lines processed in this chunk)
    """
    tmp_path, start, end, p1_mmap_path, p2_mmap_path, offset = args

    # Unpack hash table (inherited via fork COW)
    (name_buf, name_offsets, name_lengths,
     name_hashes, ids, hash_table, chain_next) = _WORKER_HASH_DATA

    # Attach to the shared memmap files
    p1_mmap = np.memmap(p1_mmap_path, dtype=np.int32, mode='r+')
    p2_mmap = np.memmap(p2_mmap_path, dtype=np.int32, mode='r+')

    actual_lines = 0
    idx = offset
    with open(tmp_path, 'rb') as f:
        # mmap the entire file (offset=0 is always page-aligned).
        # Each worker then processes only its byte range [start, end)
        # using absolute positions for .find() and adjusting to
        # local coordinates for the numpy view.
        #
        # NOTE: mmap is NOT used with a 'with' block here because
        # np.frombuffer() creates an exported buffer reference that
        # prevents mmap.__exit__ from closing. We manage cleanup
        # manually by deleting the numpy view before closing.
        full_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        buf_np = np.frombuffer(full_mmap, dtype=np.uint8)[start:end]

        pos = start
        while pos < end:
            eol = full_mmap.find(b'\n', pos)
            if eol == -1 or eol >= end:
                break
            space1 = full_mmap.find(b' ', pos)
            if space1 == -1 or space1 >= eol:
                pos = eol + 1
                continue
            space2 = full_mmap.find(b' ', space1 + 1)
            if space2 == -1 or space2 > eol:
                space2 = eol  # no score field, name2 goes to end of line

            # Convert absolute file positions to local buf_np coordinates
            local_pos = pos - start
            local_space1 = space1 - start
            local_space2 = space2 - start

            # Numba hash lookup on raw bytes — no decode, no dict
            p1_mmap[idx] = _lookup_name(
                buf_np, local_pos, local_space1 - local_pos,
                hash_table, chain_next,
                name_buf, name_offsets, name_lengths,
                name_hashes, ids,
            )
            p2_mmap[idx] = _lookup_name(
                buf_np, local_space1 + 1, local_space2 - local_space1 - 1,
                hash_table, chain_next,
                name_buf, name_offsets, name_lengths,
                name_hashes, ids,
            )
            idx += 1
            actual_lines += 1
            pos = eol + 1

        # Clean up: release numpy reference before closing the mmap
        del buf_np
        full_mmap.close()

    p1_mmap.flush()
    p2_mmap.flush()
    del p1_mmap, p2_mmap
    return actual_lines


# =====================================================================
# GPU-ACCELERATED UNIQUE (if cupy available, else numpy)
# =====================================================================

def _unique_sorted_indices(keys, force_cpu=False):
    """
    Find unique uint64 keys and their first-occurrence indices.
    
    Uses cupy GPU radix sort if available (~3s for 2.3B elements vs.
    ~90s on CPU with numpy). Falls back to numpy.unique.
    
    Note: 2.3B uint64 keys need ~55 GB on-GPU memory (input + sorted +
    workspace = 24 bytes per element). RTX 5000 Ada has 32 GB, so this
    will always fall back to CPU on that hardware.
    """
    n = len(keys)
    logger.info(f"Finding unique indices across {n:,} keys"
                f" ({n * 8 / 1e9:.1f} GB uint64)...")
    
    if not force_cpu:
        try:
            import cupy as cp
            mem_info = cp.cuda.Device().mem_info
            free_mb, total_mb = mem_info[0] / 1e6, mem_info[1] / 1e6
            # cuPy radix sort needs ~3x input size (input + sorted + workspace)
            need_gb = n * 8 * 3 / 1e9
            logger.info(f"cupy available: GPU memory {free_mb:.0f}/{total_mb:.0f} MB"
                        f", need ~{need_gb:.1f} GB")
            
            if free_mb * 0.85 < need_gb * 1e3:
                logger.warning(f"GPU memory may be insufficient"
                               f" (need ~{need_gb:.1f} GB, have {free_mb/1e3:.1f} GB)."
                               f" Falling back to CPU.")
            else:
                t0 = time.time()
                keys_gpu = cp.asarray(keys)
                logger.info(f"  Transferred to GPU in {time.time()-t0:.1f}s")
                
                t0 = time.time()
                sorted_gpu = cp.sort(keys_gpu)  # radix sort on GPU
                del keys_gpu
                logger.info(f"  GPU radix sort in {time.time()-t0:.1f}s")
                
                t0 = time.time()
                # Find unique boundaries
                diffs = sorted_gpu[1:] != sorted_gpu[:-1]
                indices = cp.where(diffs)[0] + 1
                del diffs
                result = cp.concatenate(
                    [cp.array([0], dtype=cp.int64), indices]
                ).get()
                del sorted_gpu, indices
                cp.get_default_memory_pool().free_all_blocks()
                logger.info(f"  GPU unique indices in {time.time()-t0:.1f}s")
                logger.info(f"GPU sort complete. {len(result):,} unique keys.")
                return result
                
        except ImportError:
            logger.info("cupy not installed. Using numpy CPU sort.")
        except Exception as e:
            logger.warning(f"GPU sort failed: {e}. Falling back to numpy.")
    
    # CPU fallback
    t0 = time.time()
    _, unique_indices = np.unique(keys, return_index=True)
    elapsed = time.time() - t0
    logger.info(f"CPU numpy.unique: {elapsed:.1f}s. {len(unique_indices):,} unique keys.")
    return unique_indices


# =====================================================================
# ON-DISK CACHE HELPERS
# =====================================================================

_CACHE_META_FILE = "cache_prepare_meta.json"
_CACHE_SCHEMA_VERSION = 1

def _file_sig(path):
    """Return (mtime, size) tuple for cache validation."""
    try:
        st = os.stat(path)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


def _load_meta(cache_dir):
    """Load cache metadata from JSON. Returns None on miss or schema mismatch."""
    p = Path(cache_dir) / _CACHE_META_FILE
    if p.exists():
        try:
            m = json.load(open(p, "r"))
            if m.get("schema_version") == _CACHE_SCHEMA_VERSION:
                return m
            logger.info("  Cache schema version mismatch — invalidating.")
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_meta(cache_dir_resolved, meta):
    """Save cache metadata to JSON."""
    meta["schema_version"] = _CACHE_SCHEMA_VERSION
    p = Path(cache_dir_resolved) / _CACHE_META_FILE
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)


def _build_meta(*file_paths):
    """Build metadata dict from file path strings."""
    meta = {}
    for f in file_paths:
        sig = _file_sig(f)
        if sig is not None:
            meta[str(f)] = {"mtime": sig[0], "size": sig[1]}
    return meta


def _meta_valid(meta, *file_paths):
    """Check that all file_paths match their sigs in meta."""
    for f in file_paths:
        key = str(f)
        if key not in meta:
            return False
        actual = _file_sig(f)
        if actual is None:
            return False
        expected = (meta[key]["mtime"], meta[key]["size"])
        if actual != expected:
            return False
    return True


# =====================================================================
# FASTA INDEX — SEQUENCE OFFSET / LENGTH LOOKUP
# =====================================================================

def _build_fasta_index(fasta_path):
    """
    Scan a FASTA file and build an index.
    
    Returns:
        index: dict[str, (seq_offset, seq_length)]
            seq_offset = file position of first sequence byte (after header)
            seq_length = total bytes of all sequence lines (incl. newlines)
        num_seqs: int
    """
    index = {}
    num_seqs = 0
    with open(fasta_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b'>'):
                continue
            # Extract name: ">name metadata\n" → "name"
            # Strip newline and any trailing whitespace before splitting,
            # matching the original prepare_data.py behavior.
            name = line[1:].strip().split()[0].decode('ascii')
            seq_start = f.tell()
            num_seqs += 1
            
            # Scan forward counting bytes until next '>' or EOF
            seq_length = 0
            while True:
                line_start = f.tell()
                next_line = f.readline()
                if not next_line:
                    # EOF: count remaining bytes
                    seq_length = line_start - seq_start
                    break
                if next_line.startswith(b'>'):
                    # Next header: count bytes up to this line
                    seq_length = line_start - seq_start
                    f.seek(line_start)  # rewind for next iteration
                    break
            
            index[sys.intern(name)] = (seq_start, seq_length)
    
    return index, num_seqs


# =====================================================================
# MAIN PIPELINE ENGINE
# =====================================================================

def prepare_data_fast(
    sequences_file: str,
    clusters_file: str,
    links_file: str,
    output_dir: str = ".",
    val_size: int = 250000,
    decompress_threads: int = 16,
    write_threads: int = 16,
    skip_gpu_sort: bool = False,
    keep_temp: bool = False,
    cache_dir: str = "",
    no_cache: bool = False,
) -> None:
    """
    Optimized STRING data preparation with parallel I/O and optional GPU sort.
    
    Pipeline:
        0. Decompress links.gz to temp file (parallel: rapidgzip/bgzip)
        1. Map cluster representatives (or load from cache)
        2. Multi-process mmap'd links → first pass: build ID arrays
        3. Shuffle #1 (or load surviving_file_indices from cache)
        4. Numba cluster dedup + sort GPU-accelerated
        5. Shuffle #2 + validation-leak filtering
        6a. Build FASTA index (or load from cache)
        6b. Pre-extract all needed sequences (single sequential FASTA scan)
        7. Stream outputs from mmap buffer (no random FASTA seeks)
    
    Args:
        sequences_file: Path to FASTA sequences file.
        clusters_file: Path to cluster representatives (tab-separated).
        links_file: Path to gzipped STRING links file.
        output_dir: Output directory for training/validation files.
        val_size: Number of validation links to reserve.
        decompress_threads: CPU threads for parallel gzip decompression.
        write_threads: CPU threads for parallel gzip writing.
        skip_gpu_sort: If True, force CPU numpy.unique even if cupy is available.
        keep_temp: If True, don't delete the decompressed temp file after
                   processing (useful for debugging or reuse).
        cache_dir: Directory for intermediate caches. Empty string defaults
                   to output_dir. Cached artifacts: prot_to_id.pkl,
                   prot_id_to_cluster_id.npy, surviving_file_indices.npy,
                   fasta_index.pkl.
        no_cache: If True, skip all cache checks and recompute everything.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir_resolved = Path(cache_dir if cache_dir else str(output_dir))
    cache_dir_resolved.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Step 0: Parallel decompress links.gz to a temp file on fast storage
    # ------------------------------------------------------------------
    # One decompression, two passes — eliminates the original's repeated
    # gzip decode of 2.3B lines.
    logger.info("=== Step 0: Parallel decompression ===")
    links_gz_size = os.path.getsize(links_file) / 1e9
    logger.info(f"Input: {links_file} ({links_gz_size:.1f} GB gzip)")
    
    # Put the temp file in the output directory (assumed to be on fast /scratch)
    tmp_links = output_dir / f"__decompressed_links_{os.getpid()}.tmp"
    
    _decompress_gz_parallel(links_file, tmp_links, num_threads=decompress_threads)
    tmp_size = tmp_links.stat().st_size / 1e9
    logger.info(f"Decompressed: {tmp_size:.1f} GB on {tmp_links.parent}")
    
    # ------------------------------------------------------------------
    # Step 1: Map cluster representatives (cached on disk)
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Cluster representatives ===")
    
    cache_cluster = cache_dir_resolved / "prot_to_id.pkl"
    cache_cluster_arr = cache_dir_resolved / "prot_id_to_cluster_id.npy"
    
    meta = _load_meta(cache_dir_resolved)
    
    prot_to_id = None
    prot_id_to_cluster_id = None
    
    if not no_cache and cache_cluster.exists() and cache_cluster_arr.exists():
        # Validate clusters file hasn't changed
        if meta is not None and _meta_valid(meta, clusters_file):
            logger.info("  Loading prot_to_id and prot_id_to_cluster_id from cache...")
            with open(cache_cluster, "rb") as f:
                prot_to_id = pickle.load(f)
            prot_id_to_cluster_id = np.load(cache_cluster_arr)
            logger.info(f"  Cache loaded: {len(prot_to_id):,} proteins,"
                        f" cluster array {prot_id_to_cluster_id.shape}")
        else:
            logger.info("  Cache invalid (clusters file changed or no metadata)."
                        " Recomputing.")
            meta = None  # force full recompute of cluster cache
    
    if prot_to_id is None:
        logger.info("  Parsing cluster representatives...")
        reps = {}
        with open(clusters_file, "r") as f:
            for line in tqdm(f, desc="Clusters"):
                parts = line.strip().split()
                rep, seq = parts[0], parts[1]
                reps[sys.intern(seq)] = sys.intern(rep)
        
        unique_prots = list(reps.keys())
        prot_to_id = {name: idx for idx, name in enumerate(unique_prots)}
        unique_clusters = list(set(reps.values()))
        cluster_to_id = {name: idx for idx, name in enumerate(unique_clusters)}
        prot_id_to_cluster_id = np.fromiter(
            (cluster_to_id[reps[p]] for p in unique_prots),
            dtype=np.int32, count=len(unique_prots)
        )
        
        # Save to cache
        logger.info("  Saving cluster cache...")
        with open(cache_cluster, "wb") as f:
            pickle.dump(prot_to_id, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(cache_cluster_arr, prot_id_to_cluster_id)
        
        del unique_prots, unique_clusters, cluster_to_id, reps
    
    assert prot_to_id is not None
    assert prot_id_to_cluster_id is not None
    
    # Build Numba-friendly hash table for worker processes.
    # This replaces the pickle'd dict — workers do raw-byte hash lookups
    # via Numba, avoiding 4.6B Python string allocations.
    logger.info("  Building Numba hash table for worker lookups...")
    t0 = time.time()
    global _WORKER_HASH_DATA
    _WORKER_HASH_DATA = _build_numba_hash_table(prot_to_id)
    elapsed = time.time() - t0
    logger.info(f"  Hash table built: {elapsed:.1f}s"
                f" ({len(prot_to_id):,} entries)")
    
    # ------------------------------------------------------------------
    # Step 2a: mmap decompressed links, count lines
    # ------------------------------------------------------------------
    logger.info("=== Step 2: First pass — multi-process ID mapping ===")
    # Memory-map the decompressed file (zero-copy, supports .find() on bytes)
    with open(tmp_links, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    
    buf_view = np.frombuffer(buf, dtype=np.uint8)
    total_links = 2_365_172_266  # magic number for STRINGDB v12 (kept as reference)
    
    # Find header end
    header_end = buf.find(b'\n')
    if header_end == -1:
        raise ValueError("Empty links file")
    logger.info(f"Total data links (reference): {total_links:,}")
    
    n = len(buf)
    
    # Count actual lines in the data portion (after header)
    #data_view = buf_view[header_end + 1:]
    #actual_line_count = np.count_nonzero(data_view == ord('\n'))
    #logger.info(f"  Actual data lines (from buffer scan): {actual_line_count:,}")
    
    # Use actual count, fall back to magic number only if buffer scan seems wrong
    #if actual_line_count > 0:
    #    total_links = actual_line_count
    
    # ------------------------------------------------------------------
    # Step 2b: Multi-process link → ID mapping via memmap segments
    # ------------------------------------------------------------------
    # Split the mmap'd buffer into chunks at newline boundaries. Each worker
    # gets a byte range, loads prot_to_id from the cached pickle, and writes
    # results into a shared memmap array. This avoids Python pickle overhead
    # of transferring 18 GB of arrays back through the pipe.
    
    num_workers = min(int(os.cpu_count() or 4), 24, max(1, int(total_links // 10_000_000)))
    logger.info(f"  Using {num_workers} worker processes")
    
    # Find chunk boundaries at newlines
    chunk_boundaries = [header_end + 1]
    chunk_size = n // num_workers
    for i in range(1, num_workers):
        boundary = i * chunk_size
        if boundary <= header_end:
            continue
        eol = buf.find(b'\n', int(boundary))
        if eol == -1 or eol >= n - 1:
            break
        chunk_boundaries.append(eol + 1)
    if chunk_boundaries[-1] != n:
        chunk_boundaries.append(n)
    
    # Count actual lines per chunk using fast numpy vectorized scan
    chunk_line_counts = []
    for i in range(len(chunk_boundaries) - 1):
        chunk_slice = buf_view[chunk_boundaries[i]:chunk_boundaries[i+1]]
        chunk_line_counts.append(np.count_nonzero(chunk_slice == ord('\n')))
    # chunk_slice is a live NumPy view of buf_view — release it now so it
    # doesn't hold an exported pointer on the mmap buffer until the end.
    del chunk_slice
    
    # Compute exact array offsets for each chunk
    exact_offsets = [0]
    for c in chunk_line_counts[:-1]:
        exact_offsets.append(exact_offsets[-1] + c)
    total_data_lines = sum(chunk_line_counts)
    
    logger.info(f"  Confirmed {total_data_lines:,} data lines from chunk line counts")
    
    # Create shared memmap files for output arrays
    p1_mmap_path = os.path.join(
        str(tmp_links.parent), f"__p1_ids_{os.getpid()}.mmap")
    p2_mmap_path = os.path.join(
        str(tmp_links.parent), f"__p2_ids_{os.getpid()}.mmap")
    
    p1_mmap = np.memmap(p1_mmap_path, dtype=np.int32, mode='w+',
                        shape=(total_data_lines,))
    p2_mmap = np.memmap(p2_mmap_path, dtype=np.int32, mode='w+',
                        shape=(total_data_lines,))
    # Force initial page-in (sequential write)
    p1_mmap[0] = 0
    p2_mmap[0] = 0
    p1_mmap.flush()
    p2_mmap.flush()
    
    # Build chunk argument list
    chunk_args = []
    for i in range(len(chunk_boundaries) - 1):
        chunk_args.append((
            str(tmp_links),
            chunk_boundaries[i], chunk_boundaries[i+1],
            p1_mmap_path, p2_mmap_path,
            exact_offsets[i],
        ))
    
    # Dispatch workers
    t0 = time.time()
    ctx = mp.get_context('fork')
    with ctx.Pool(num_workers) as pool:
        counts = pool.map(_worker_process_chunk, chunk_args)
    elapsed = time.time() - t0
    
    actual_total = sum(counts)
    logger.info(f"  Multi-process mapping: {elapsed:.1f}s, {actual_total:,} links mapped")
    
    # Load into regular memory, then free the memmap files
    trimmed = False
    if actual_total < total_data_lines:
        p1_ids = np.array(p1_mmap[:actual_total])
        p2_ids = np.array(p2_mmap[:actual_total])
        total_links = actual_total
        trimmed = True
        logger.info(f"  Trimmed to {total_links:,} valid links")
    else:
        p1_ids = np.array(p1_mmap)
        p2_ids = np.array(p2_mmap)
    
    # Clean up memmap files
    del p1_mmap, p2_mmap
    try:
        os.unlink(p1_mmap_path)
        os.unlink(p2_mmap_path)
    except OSError:
        pass
    
    # ------------------------------------------------------------------
    # Step 3: Shuffle #1 + Step 4: Cluster dedup (or load from cache)
    # ------------------------------------------------------------------
    cache_surviving = cache_dir_resolved / "surviving_file_indices.npy"
    
    # Rebuild meta with links info if needed
    if meta is None:
        meta = _build_meta(clusters_file, links_file, sequences_file)
        meta["total_links"] = total_links
        meta["trimmed"] = trimmed
    
    use_surviving_cache = (
        not no_cache
        and cache_surviving.exists()
        and meta is not None
        and _meta_valid(meta, clusters_file, links_file)
        and meta.get("total_links") == total_links
        and not trimmed  # can't trust cache if trimming changed total_links
    )
    
    if use_surviving_cache:
        logger.info("=== Step 3+4: Loading surviving_file_indices from cache ===")
        surviving_file_indices = np.load(cache_surviving)
        logger.info(f"  Loaded: {len(surviving_file_indices):,} surviving links"
                    f" ({100 * len(surviving_file_indices) / total_links:.1f}%)")
    else:
        logger.info("=== Step 3: Shuffle 1 ===")
        line_order = np.arange(total_links, dtype=np.int64)
        rng = np.random.default_rng(137)
        rng.shuffle(line_order)
        
        p1_shuffled1 = p1_ids[line_order]
        p2_shuffled1 = p2_ids[line_order]
        # Keep p1_ids, p2_ids alive — needed in Step 5 for leak filtering
        
        # Step 4: Cluster dedup with GPU-accelerated sort
        logger.info("=== Step 4: Unique cluster links ===")
        pair_keys = generate_pair_keys_numba(
            p1_shuffled1, p2_shuffled1, prot_id_to_cluster_id
        )
        # Can now free the shuffled copies (originals p1_ids/p2_ids stay)
        del p1_shuffled1, p2_shuffled1
        
        # GPU-accelerated radix sort (or numpy fallback)
        unique_indices = _unique_sorted_indices(pair_keys, force_cpu=skip_gpu_sort)
        del pair_keys
        unique_indices.sort()
        
        surviving_file_indices = line_order[unique_indices]
        del unique_indices, line_order
        logger.info(f"Kept {len(surviving_file_indices):,} / {total_links:,} links"
                    f" ({100 * len(surviving_file_indices) / total_links:.1f}%)")
        
        # Save to cache
        logger.info("  Saving surviving_file_indices cache...")
        np.save(cache_surviving, surviving_file_indices)
    
    _save_meta(cache_dir_resolved, meta)
    
    # ------------------------------------------------------------------
    # Step 5: Shuffle #2 + validation-leak filtering
    # ------------------------------------------------------------------
    logger.info("=== Step 5: Train/Val split ===")
    rng_split = np.random.default_rng(731)
    rng_split.shuffle(surviving_file_indices)
    
    val_file_indices = surviving_file_indices[:val_size]
    train_file_indices = surviving_file_indices[val_size:]
    del surviving_file_indices
    
    # Build validation cluster mask from the ORIGINAL p1_ids/p2_ids
    val_clus_mask = np.zeros(len(prot_id_to_cluster_id), dtype=np.bool_)
    val_clus_mask[prot_id_to_cluster_id[p1_ids[val_file_indices]]] = True
    val_clus_mask[prot_id_to_cluster_id[p2_ids[val_file_indices]]] = True
    
    # Filter training indices via Numba parallel kernel
    filtered_train_file_indices = filter_leaking_train_numba(
        p1_ids, p2_ids, train_file_indices,
        val_clus_mask, prot_id_to_cluster_id,
    )
    del train_file_indices, val_clus_mask
    
    # Create O(1) boolean masks for the streaming pass
    is_val = np.zeros(total_links, dtype=bool)
    is_val[val_file_indices] = True
    del val_file_indices
    
    is_train_filtered = np.zeros(total_links, dtype=bool)
    is_train_filtered[filtered_train_file_indices] = True
    del filtered_train_file_indices
    
    # Can now free p1_ids, p2_ids, and prot_to_id/prot_id_to_cluster_id
    del p1_ids, p2_ids, prot_to_id, prot_id_to_cluster_id
    
    # ------------------------------------------------------------------
    # Step 6a: Build or load FASTA index (needed for sequence offsets)
    # ------------------------------------------------------------------
    cache_fasta = cache_dir_resolved / "fasta_index.pkl"
    
    if not no_cache and cache_fasta.exists():
        # Validate sequences file hasn't changed
        if _meta_valid(meta, sequences_file):
            logger.info("=== Step 6a: Loading FASTA index from cache ===")
            t0 = time.time()
            with open(cache_fasta, "rb") as f:
                fasta_index = pickle.load(f)
            elapsed = time.time() - t0
            logger.info(f"  FASTA index loaded: {len(fasta_index):,} sequences"
                        f" ({elapsed:.1f}s)")
        else:
            logger.info("=== Step 6a: FASTA file changed, rebuilding index ===")
            fasta_index, _ = _build_fasta_index(sequences_file)
            _save_meta(cache_dir_resolved, meta)
    else:
        logger.info("=== Step 6a: Building FASTA index ===")
        fasta_index, _ = _build_fasta_index(sequences_file)
        # Save index to cache
        logger.info("  Saving FASTA index cache...")
        t0 = time.time()
        with open(cache_fasta, "wb") as f:
            pickle.dump(fasta_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed = time.time() - t0
        logger.info(f"  FASTA index cached ({elapsed:.1f}s)")
    
    # ------------------------------------------------------------------
    # Step 6b: Pre-extract all needed sequences (single sequential FASTA scan)
    # ------------------------------------------------------------------
    # Instead of lazy random-access FASTA reading during the streaming pass
    # (which causes random seeks and requires an LRU cache), we collect the
    # set of unique protein names that appear in val/train links, then scan
    # the FASTA file once sequentially.
    
    logger.info("=== Step 6b: Pre-extracting needed sequences ===")
    needed_names = set()
    pos = header_end + 1
    for idx in range(total_links):
        eol = buf.find(b'\n', pos)
        if eol == -1:
            line = buf[pos:]
            pos = n
        else:
            line = buf[pos:eol]
            pos = eol + 1
        
        if not is_val[idx] and not is_train_filtered[idx]:
            continue
        
        # Parse the line — format: name1 name2 [score]
        line = line.decode('ascii')
        parts = line.split()
        name1 = parts[0]
        name2 = parts[1]
        
        #space1 = line.find(b' ')
        #if space1 == -1:
        #    continue
        #space2 = line.find(b' ', space1 + 1)
        #if space2 == -1:
        #    space2 = len(line)  # no score, name2 goes to end
        #name1 = line[:space1].decode('ascii')
        #name2 = line[space1 + 1:space2].decode('ascii')
        needed_names.add(name1)
        needed_names.add(name2)
    
    logger.info(f"  {len(needed_names):,} unique protein sequences needed")
    
    # Look up each needed name in the FASTA index and collect (offset, name, length)
    # tuples, then sort by offset for a single sequential pass. This is more robust
    # than iterating fasta_index.items() and checking membership, because it works
    # regardless of dict iteration order and name-interning differences.
    needed_with_offsets = []
    for name in needed_names:
        entry = fasta_index.get(name)
        if entry is not None:
            needed_with_offsets.append((entry[0], name, entry[1]))
    needed_with_offsets.sort()  # sort by file offset → sequential reads
    
    unfound = len(needed_names) - len(needed_with_offsets)
    if unfound:
        logger.warning(f"  {unfound:,} needed sequences not found in FASTA index!")
    del needed_names
    
    # Single sequential pass through the FASTA file
    needed_seqs = {}
    with open(sequences_file, 'rb') as f:
        for offset, name, length in tqdm(needed_with_offsets,
                                          desc="Reading FASTA", miniters=10_000):
            f.seek(offset)
            raw = f.read(length)
            needed_seqs[name] = raw.replace(b'\n', b'').decode('ascii')
    
    logger.info(f"  Retrieved {len(needed_seqs):,} sequences")
    
    # Free the FASTA index (not needed after scanning)
    del fasta_index
    
    # ------------------------------------------------------------------
    # Step 7: Streaming outputs (parallel zstd)
    # ------------------------------------------------------------------
    logger.info("=== Step 7: Streaming outputs (parallel zstd) ====")
    written_val_seqs = set()
    written_filtered_seqs = set()
    
    # Open output files via multithreaded zstd
    f_v_lnk, close_v_lnk = _open_zst_write(output_dir / "validation.links.txt.zst",
                                           write_threads)
    f_v_seq, close_v_seq = _open_zst_write(output_dir / "validation.seqs.txt.zst",
                                           write_threads)
    f_tf_lnk, close_tf_lnk = _open_zst_write(
        output_dir / "training_filtered.links.txt.zst", write_threads)
    f_tf_seq, close_tf_seq = _open_zst_write(
        output_dir / "training_filtered.seqs.txt.zst", write_threads)
    
    # Re-scan the decompressed buffer for the second pass
    num_train_link = 0
    pos = header_end + 1
    PROGRESS_BATCH = 1_000_000
    with tqdm(total=total_links, desc="Streaming outputs") as pbar:
        next_progress = PROGRESS_BATCH
        for idx in range(total_links):
            # Read next line
            eol = buf.find(b'\n', pos)
            if eol == -1:
                line = buf[pos:]
                pos = n
            else:
                line = buf[pos:eol]
                pos = eol + 1
            
            # Fast path: skip if not needed
            if not is_val[idx] and not is_train_filtered[idx]:
                if idx + 1 >= next_progress:
                    pbar.update(PROGRESS_BATCH)
                    next_progress += PROGRESS_BATCH
                continue
            
            # Parse the line — format: name1 name2 [score]
            line = line.decode('ascii')
            parts = line.split()
            name1 = parts[0]
            name2 = parts[1]
            
            if is_val[idx]:
                f_v_lnk.write(f"{name1} {name2}\n")
                if name1 not in written_val_seqs:
                    seq = needed_seqs.get(name1)
                    if seq is not None:
                        f_v_seq.write(f"{name1} {seq}\n")
                        written_val_seqs.add(name1)
                if name2 not in written_val_seqs:
                    seq = needed_seqs.get(name2)
                    if seq is not None:
                        f_v_seq.write(f"{name2} {seq}\n")
                        written_val_seqs.add(name2)
            elif is_train_filtered[idx]:
                f_tf_lnk.write(f"{name1} {name2}\n")
                if name1 not in written_filtered_seqs:
                    seq = needed_seqs.get(name1)
                    if seq is not None:
                        f_tf_seq.write(f"{name1} {seq}\n")
                        written_filtered_seqs.add(name1)
                if name2 not in written_filtered_seqs:
                    seq = needed_seqs.get(name2)
                    if seq is not None:
                        f_tf_seq.write(f"{name2} {seq}\n")
                        written_filtered_seqs.add(name2)
                num_train_link += 1
            if idx + 1 >= next_progress:
                pbar.update(PROGRESS_BATCH)
                next_progress += PROGRESS_BATCH
        pbar.n = total_links
        pbar.refresh()
    
    # Close all zstd writers properly
    close_v_lnk()
    close_v_seq()
    close_tf_lnk()
    close_tf_seq()
    
    # Clean up
    # buf_view (np.frombuffer wrapper from Step 2a) still holds an exported
    # pointer to the mmap buffer — must delete it before closing the mmap.
    del buf_view
    buf.close()
    
    if not keep_temp:
        tmp_links.unlink(missing_ok=True)
        logger.info(f"Temp file removed: {tmp_links}")
    
    logger.info(f"Finished! {num_train_link:,} training links,"
                f" {val_size:,} validation links.")


def _decompress_gz_parallel(gz_path, out_path, num_threads=16):
    """
    Decompress a gzip file to a plain-text file using the best available
    parallel tool.
    
    Tool preference:
      1. rapidgzip — parallel decompression of any gzip stream
      2. bgzip     — block-gzip block-parallel
      3. pigz -d   — multi-stream pigz decompression
      4. Python gzip — single-threaded fallback
    """
    gz_path = Path(gz_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    
    for tool_name, cmd in [
        ("rapidgzip",
         ["rapidgzip", "-d", "-P", str(num_threads), "-f", str(gz_path),
          "-o", str(out_path)]),
        ("bgzip",
         ["bgzip", "-d", "-p", str(num_threads), "-c", str(gz_path)]),
        ("pigz",
         ["pigz", "-d", "-p", str(num_threads), "-c", str(gz_path)]),
    ]:
        exe = _find_tool(tool_name)
        if exe is None:
            continue
        
        cmd[0] = exe  # use full path
        
        try:
            if tool_name == "rapidgzip":
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True)
            else:
                # bgzip / pigz write to stdout
                with open(out_path, "wb") as f:
                    result = subprocess.run(
                        cmd, stdout=f, stderr=subprocess.PIPE, check=True)
            
            elapsed = time.time() - t0
            logger.info(f"{tool_name}: {elapsed:.1f}s ({num_threads} threads)")
            return
            
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            logger.warning(f"{tool_name} failed (exit={e.returncode}):"
                           f" {e.stderr if e.stderr else ''}")
            continue
    
    # Python fallback
    logger.warning("No parallel decompression tool found."
                   " Using single-threaded Python gzip."
                   " Install rapidgzip for ~4x speedup.")
    t0 = time.time()
    with _gzip_backend.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    elapsed = time.time() - t0
    logger.info(f"Python gzip decompression: {elapsed:.1f}s")


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare STRING data for MINT+ training (optimized).",
    )
    parser.add_argument("--sequences", required=True,
                        help="Path to FASTA sequences file")
    parser.add_argument("--clusters", required=True,
                        help="Path to cluster representatives file")
    parser.add_argument("--links", required=True,
                        help="Path to gzipped STRING links file")
    parser.add_argument("--output-dir", default="./data/diamond",
                        help="Output directory (default: ./data/diamond)")
    parser.add_argument("--val-size", type=int, default=250000,
                        help="Number of validation links (default: 250000)")
    parser.add_argument("--decompress-threads", type=int, default=16,
                        help="Threads for parallel gzip decompression")
    parser.add_argument("--write-threads", type=int, default=8,
                        help="Threads for parallel gzip writing")
    parser.add_argument("--cpu-sort", action="store_true",
                        help="Force CPU numpy sort (skip GPU even if cupy available)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep decompressed temp file after processing")
    parser.add_argument("--cache-dir", default="",
                        help="Directory for intermediate caches"
                             " (default: same as --output-dir)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip all cache checks, recompute everything")
    
    args = parser.parse_args()
    
    prepare_data_fast(
        sequences_file=args.sequences,
        clusters_file=args.clusters,
        links_file=args.links,
        output_dir=args.output_dir,
        val_size=args.val_size,
        decompress_threads=args.decompress_threads,
        write_threads=args.write_threads,
        skip_gpu_sort=args.cpu_sort,
        keep_temp=args.keep_temp,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    )
