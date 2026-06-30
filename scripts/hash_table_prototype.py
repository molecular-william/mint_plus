#!/usr/bin/env python3
"""
Prototype: Numba-accelerated protein-name-to-ID lookup.

Replaces Python dict lookup + .decode('ascii') with a hash table
implemented in Numba, operating directly on raw bytes. This avoids
allocating 4.6 billion Python strings in the first pass (2 names x
2.3B lines).

Data structures (all flat NumPy arrays, Numba-friendly):
  name_buf       — uint8[N_bytes], all names concatenated
  name_offsets   — int64[N], byte offset of each name in name_buf
  name_lengths   — int32[N], byte length of each name
  name_hashes    — uint32[N], DJB2 hash of each name
  ids            — int32[N], the integer ID for each name
  hash_table     — int32[T], slots = next_pow2(2*N), -1 = empty
  chain_next     — int32[N], collision chain link (-1 = end)

Lookup in Numba:
  1. Compute DJB2 hash of the raw name bytes
  2. Index into hash_table by (hash & mask)
  3. Walk the chain, comparing full hash + length first, then bytes
  4. Return the integer ID

Usage:
    python scripts/hash_table_prototype.py
        Optional: --num-names N  (default: 2.3M, matching STRING v12)
        Optional: --num-lookups N (default: 10M)
        Optional: --synthetic     (default: generates STRING-like names)
"""

import sys
import time
import argparse
import random
import string

import numpy as np
import numba as nb


# =====================================================================
# HASH FUNCTION (Numba-compatible, operates on raw bytes)
# =====================================================================

@nb.njit
def _djb2_hash(buf, start, length):
    """Compute DJB2 hash of buf[start:start+length].
    
    Uses explicit 32-bit masking to ensure wrapping on both Numba
    and Python sides.
    """
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


# =====================================================================
# HASH TABLE LOOKUP (Numba)
# =====================================================================

@nb.njit
def lookup_name(buf_bytes, start, length,
                hash_table, chain_next,
                name_buf, name_offsets, name_lengths,
                name_hashes, ids):
    """
    Look up a protein name from raw bytes.
    
    Returns the integer ID, or -1 if not found.
    
    buf_bytes  : uint8[] — the buffer containing the name
    start      : where the name starts in buf_bytes
    length     : byte length of the name
    hash_table : int32[T] — hash slot heads
    ...
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


# =====================================================================
# BUILD THE HASH TABLE (from Python dict)
# =====================================================================

def build_numba_hash_table(prot_to_id):
    """
    Build Numba-friendly hash table from a Python dict.
    
    Args:
        prot_to_id: dict[str, int] — mapping of protein name to integer ID
    
    Returns:
        (name_buf, name_offsets, name_lengths, name_hashes,
         ids, hash_table, chain_next, table_mask)
    """
    num_prots = len(prot_to_id)
    
    # Pre-encode and hash all names
    name_items = []
    for name, pid in prot_to_id.items():
        name_bytes = name.encode('ascii')
        h = _djb2_hash(np.frombuffer(name_bytes, dtype=np.uint8), 0, len(name_bytes))
        name_items.append((name_bytes, h, pid))
    
    # Sort names alphabetically — not required for the hash table,
    # but ensures deterministic ordering of chain entries.
    name_items.sort(key=lambda x: x[0])
    
    # Concatenate all names into one buffer
    name_parts = [item[0] for item in name_items]
    name_buf = np.frombuffer(b''.join(name_parts), dtype=np.uint8)
    
    # Arrays
    name_offsets = np.empty(num_prots, dtype=np.int64)
    name_lengths = np.empty(num_prots, dtype=np.int32)
    name_hashes = np.empty(num_prots, dtype=np.uint32)
    ids = np.empty(num_prots, dtype=np.int32)
    chain_next = np.empty(num_prots, dtype=np.int32)
    
    # Determine hash table size (next power of 2 >= 2 * num_prots)
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
        
        # Insert into chain
        slot = h & (table_size - 1)
        chain_next[i] = hash_table[slot]
        hash_table[slot] = i
    
    return (name_buf, name_offsets, name_lengths, name_hashes,
            ids, hash_table, chain_next)


# =====================================================================
# GENERATE SYNTHETIC PROTEIN NAMES (STRING-like)
# =====================================================================

def generate_synthetic_proteins(num_names, random_seed=42):
    """
    Generate synthetic protein names resembling STRING-DB identifiers.
    
    Format: "9606.ENSP" + 11 random alphanumeric characters
    Example: "9606.ENSP00000288674"
    """
    rng = random.Random(random_seed)
    # Prefix pool to create realistic-looking IDs
    prefixes = [
        "9606.ENSP",  # human
        "10090.ENSMUSP",  # mouse
        "7955.ENSDARP",  # zebrafish
        "7227.ENSP",  # fly
        "6239.ENSP",  # worm
        "3702.ENSP",  # arabidopsis
        "83333.ENSP",  # E. coli
        "284812.ENSP",  # yeast
    ]
    
    prot_to_id = {}
    seen = set()
    chars = string.digits  # STRING IDs are numeric
    
    while len(prot_to_id) < num_names:
        prefix = rng.choice(prefixes)
        suffix = ''.join(rng.choices(chars, k=11))
        name = f"{prefix}{suffix}"
        if name not in seen:
            seen.add(name)
            prot_to_id[name] = len(prot_to_id)
    
    return prot_to_id


# =====================================================================
# BENCHMARK: Python dict vs Numba hash table
# =====================================================================

def run_benchmark(prot_to_id, num_lookups, batch_size=100000):
    """
    Benchmark Python dict vs Numba hash table for protein name lookup.
    
    Uses synthetic queries (a mix of existing + non-existing names).
    
    Returns:
        (dict_time, hash_time, correct_dict, correct_hash)
    """
    all_names = list(prot_to_id.keys())
    num_prots = len(all_names)
    
    # Build hash table
    name_buf, name_offsets, name_lengths, name_hashes, ids, hash_table, chain_next = \
        build_numba_hash_table(prot_to_id)
    
    # Generate random lookup queries:
    # ~90% existing names, ~10% non-existing (simulating real data where
    # most lines are valid, but some may reference unknown proteins)
    rng = random.Random(123)
    
    # Pre-generate as bytes for a fair comparison
    query_names = []
    for _ in range(num_lookups):
        if rng.random() < 0.9:
            name = rng.choice(all_names)
        else:
            # Generate a non-existent name
            prefix = "9606.ENSP"
            suffix = ''.join(rng.choices(string.digits, k=11))
            name = f"{prefix}{suffix}"
        query_names.append(name)
    
    # Also pre-encode to bytes
    query_bytes = [n.encode('ascii') for n in query_names]
    
    # ---- Python dict benchmark ----
    t0 = time.perf_counter()
    results_dict = []
    for name in query_names:
        results_dict.append(prot_to_id.get(name, -1))
    dict_time = time.perf_counter() - t0
    
    # ---- Numba hash table benchmark ----
    # We need to call the Numba function for each query.
    # Each call passes the name as a uint8 view of the bytes buffer.
    # To minimize call overhead, batch the lookups: pass many names at once.
    
    # Create a single buffer with all query names concatenated
    q_offsets = np.empty(num_lookups + 1, dtype=np.int64)
    q_lengths = np.empty(num_lookups, dtype=np.int32)
    
    buf_offset = 0
    q_parts = []
    for i, b in enumerate(query_bytes):
        q_offsets[i] = buf_offset
        q_lengths[i] = len(b)
        q_parts.append(b)
        buf_offset += len(b)
    q_offsets[num_lookups] = buf_offset
    
    q_buf = np.frombuffer(b''.join(q_parts), dtype=np.uint8)
    
    # Numba-compile the batch lookup
    @nb.njit(parallel=True)
    def batch_lookup(q_buf, q_offsets, q_lengths,
                     hash_table, chain_next,
                     name_buf, name_offsets, name_lengths,
                     name_hashes, ids):
        n = len(q_lengths)
        out = np.empty(n, dtype=np.int32)
        for i in nb.prange(n):
            start = q_offsets[i]
            length = q_lengths[i]
            out[i] = lookup_name(
                q_buf, start, length,
                hash_table, chain_next,
                name_buf, name_offsets, name_lengths,
                name_hashes, ids,
            )
        return out
    
    # Warm-up: compile the Numba kernel
    _ = batch_lookup(q_buf[:1], q_offsets[:2], q_lengths[:1],
                     hash_table, chain_next,
                     name_buf, name_offsets, name_lengths,
                     name_hashes, ids)
    
    t0 = time.perf_counter()
    results_hash = batch_lookup(
        q_buf, q_offsets, q_lengths,
        hash_table, chain_next,
        name_buf, name_offsets, name_lengths,
        name_hashes, ids,
    )
    hash_time = time.perf_counter() - t0
    
    # Verify correctness
    correct_dict = sum(1 for r in results_dict if r >= 0)
    correct_hash = sum(1 for r in results_hash if r >= 0)
    all_match = all(a == b for a, b in zip(results_dict, results_hash))
    
    print(f"  Name buffer size: {len(name_buf):,} bytes "
          f"({len(name_buf) / 1e6:.1f} MB)")
    print(f"  Hash table size: {len(hash_table):,} slots "
          f"(load factor: {num_prots / len(hash_table):.2f})")
    print(f"  Hash chain stats:")
    print(f"    Empty slots: {np.count_nonzero(hash_table == -1):,} "
          f"({100 * np.count_nonzero(hash_table == -1) / len(hash_table):.1f}%)")
    
    chain_lengths = []
    for slot in range(len(hash_table)):
        idx = hash_table[slot]
        length = 0
        while idx != -1:
            length += 1
            idx = chain_next[idx]
        if length > 0:
            chain_lengths.append(length)
    if chain_lengths:
        print(f"    Non-empty chains: {len(chain_lengths)}")
        print(f"    Max chain length: {max(chain_lengths)}")
        print(f"    Mean chain length: {np.mean(chain_lengths):.2f}")
    
    return dict_time, hash_time, correct_dict, correct_hash, all_match


# =====================================================================
# DEMONSTRATE SINGLE-LOOKUP PATTERN (how workers would use it)
# =====================================================================

def demonstrate_worker_pattern(prot_to_id):
    """
    Show how a worker would use the Numba lookup in the hot loop,
    without going through Python's decode + dict.
    
    This simulates the pattern:
      line = buf[pos:eol]           # raw bytes from mmap
      space = line.find(b' ')
      name1 = line[:space]
      name2 = line[space+1:]
      p1[idx] = lookup_name(name1)  # no decode, no dict
    """
    print("\n--- Worker Integration Pattern ---")
    
    name_buf, name_offsets, name_lengths, name_hashes, ids, hash_table, chain_next = \
        build_numba_hash_table(prot_to_id)
    
    # Grab a sample name
    sample_name = next(iter(prot_to_id.keys()))
    sample_bytes = sample_name.encode('ascii')
    expected_id = prot_to_id[sample_name]
    
    # Simulate what a worker sees: raw bytes from mmap
    line = f"{sample_name} 9606.ENSP00000000000".encode('ascii')
    space = line.find(b' ')
    name1_bytes = line[:space]
    name2_bytes = line[space + 1:]
    
    # Convert to uint8 view (zero-copy)
    buf_view = np.frombuffer(line, dtype=np.uint8)
    
    # Lookup name1 via Numba
    found_id = lookup_name(
        buf_view, 0, len(name1_bytes),  # name1 starts at 0
        hash_table, chain_next,
        name_buf, name_offsets, name_lengths,
        name_hashes, ids,
    )
    
    # Lookup name2 via Numba (should return -1 — not in dict)
    not_found = lookup_name(
        buf_view, space + 1, len(name2_bytes),
        hash_table, chain_next,
        name_buf, name_offsets, name_lengths,
        name_hashes, ids,
    )
    
    status = "PASS" if found_id == expected_id and not_found == -1 else "FAIL"
    print(f"  [{status}] lookup_name('{sample_name}') = {found_id} "
          f"(expected {expected_id})")
    print(f"  [{status}] lookup_name(non-existent) = {not_found} "
          f"(expected -1)")
    print(f"  Decode+dict avoided: {len(name1_bytes)} bytes "
          f"→ Numba hash directly")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prototype Numba-friendly protein name lookup",
    )
    parser.add_argument("--num-names", type=int, default=2_300_000,
                        help="Number of protein names in the mapping "
                             "(default: 2.3M, matching STRING v12)")
    parser.add_argument("--num-lookups", type=int, default=10_000_000,
                        help="Number of random lookups for benchmark "
                             "(default: 10M)")
    parser.add_argument("--batch-size", type=int, default=100_000,
                        help="Lookup batch size (default: 100k)")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip benchmark, only build and verify")
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic names (default: True)")
    
    args = parser.parse_args()
    
    print(f"=== Numba Hash Table Prototype ===")
    print(f"Protein names: {args.num_names:,}")
    print(f"Lookups: {args.num_lookups:,}")
    
    # Generate synthetic mapping
    print("\n--- Generating synthetic protein names ---")
    t0 = time.time()
    prot_to_id = generate_synthetic_proteins(args.num_names)
    gen_time = time.time() - t0
    print(f"  Generated {len(prot_to_id):,} names in {gen_time:.2f}s")
    print(f"  Example: {next(iter(prot_to_id.keys()))}")
    
    # Demo single-lookup pattern
    demonstrate_worker_pattern(prot_to_id)
    
    # Build hash table and benchmark
    print("\n--- Building Numba Hash Table ---")
    t0 = time.time()
    name_buf, name_offsets, name_lengths, name_hashes, ids, hash_table, chain_next = \
        build_numba_hash_table(prot_to_id)
    build_time = time.time() - t0
    print(f"  Built in {build_time:.2f}s")
    
    if not args.no_benchmark:
        print("\n--- Benchmark: Dict vs Numba Hash ---")
        dict_time, hash_time, correct_dict, correct_hash, all_match = run_benchmark(
            prot_to_id, args.num_lookups, args.batch_size,
        )
        
        # Corrections per sec
        dict_cps = args.num_lookups / dict_time
        hash_cps = args.num_lookups / hash_time
        speedup = hash_cps / dict_cps
        
        print(f"\n  {'Method':<20} {'Time (s)':<12} {'Lookups/s':<16} {'Found':<10}")
        print(f"  {'-'*20} {'-'*12} {'-'*16} {'-'*10}")
        print(f"  {'Python dict':<20} {dict_time:<12.2f} {dict_cps:<16,.0f} {correct_dict:<10,}")
        print(f"  {'Numba hash table':<20} {hash_time:<12.2f} {hash_cps:<16,.0f} {correct_hash:<10,}")
        print(f"\n  Results match: {'YES' if all_match else 'NO'}")
        print(f"  Numba speedup vs dict: {speedup:.1f}x")
        
        # At 2.3B lines (each link has 2 names = 4.6B lookups)
        dict_total = 4.6e9 / dict_cps / 3600
        hash_total = 4.6e9 / hash_cps / 3600
        print(f"\n  Projected total time for 2.3B lines (4.6B lookups):")
        print(f"    Python dict: {dict_total:.1f} hours")
        print(f"    Numba hash:  {hash_total:.1f} hours")
    
    print("\n--- Memory Footprint ---")
    total_bytes = (
        name_buf.nbytes
        + name_offsets.nbytes
        + name_lengths.nbytes
        + name_hashes.nbytes
        + ids.nbytes
        + hash_table.nbytes
        + chain_next.nbytes
    )
    print(f"  name_buf:       {name_buf.nbytes / 1e6:.1f} MB")
    print(f"  name_offsets:   {name_offsets.nbytes / 1e6:.1f} MB")
    print(f"  name_lengths:   {name_lengths.nbytes / 1e6:.1f} MB")
    print(f"  name_hashes:    {name_hashes.nbytes / 1e6:.1f} MB")
    print(f"  ids:            {ids.nbytes / 1e6:.1f} MB")
    print(f"  hash_table:     {hash_table.nbytes / 1e6:.1f} MB")
    print(f"  chain_next:     {chain_next.nbytes / 1e6:.1f} MB")
    print(f"  Total:          {total_bytes / 1e6:.1f} MB")
    
    print("\nDone.")
