# GPU Hash Set Implementation

This repository contains a high-performance GPU-based hash set implementation using HIP (AMD's GPU programming platform). The hash set uses open addressing with linear probing for collision resolution and is designed for parallel insertion of elements.

## Overview

The hash set is implemented as a fixed-size array of buckets, where each bucket can store multiple elements. This design is optimized for GPU parallel processing by allowing concurrent insertions while managing conflicts efficiently.

### Key Features

- Parallel insertion using GPU threads
- Open addressing with linear probing for collision handling
- Warp-aligned bucket size (64 elements per bucket) for optimal GPU performance
- Cooperative group-based implementation for efficient thread collaboration
- Configurable load factor to balance memory usage and performance
- Memory-efficient design with single-thread atomic operations

## Implementation Details

### Data Structure

The hash set consists of:
- An array of buckets, where each bucket can store `bucket_size` elements (64 elements, matching GPU warp size)
- A sentinel value to mark empty slots
- A configurable load factor to determine the total number of buckets

### Key Components

1. **Hash Function**
   ```cpp
   uint32_t hash_x = 0x9e3779b9;
   uint32_t hash_y = 0x9e7019b9;
   const uint32_t prime_divisor = 1 << 31;
   const auto hash = (((hash_x ^ key_to_insert) + hash_y) % prime_divisor);
   ```
   The implementation uses a simple hash function combining XOR operations and prime numbers. The constants are chosen to provide good distribution properties.

2. **Bucket Management**
   - Each bucket can store up to `bucket_size` elements (64, matching GPU warp size)
   - Buckets are processed in parallel by thread blocks
   - Linear probing is used when a bucket is full
   - The bucket size is aligned with GPU warp size for optimal performance

3. **Parallel Insertion**
   - Uses cooperative groups for thread coordination
   - Thread tiles of size `bucket_size` work together on insertions
   - Single-thread atomic operations minimize memory contention
   - The thread that loaded the key performs the insertion (same lane) to avoid unnecessary value broadcasting
   - Only one thread per warp performs the actual insertion to reduce unnecessary memory operations

### Insertion Algorithm

1. Each thread loads a key to insert
2. The hash value is computed to determine the initial bucket
3. Within each thread tile (warp):
   - The thread that loaded the key performs the insertion (same lane) to avoid value broadcasting
   - The thread attempts to find an empty slot using atomic operations
   - If the bucket is full, linear probing moves to the next bucket
   - Process continues until insertion succeeds or all buckets are checked
   - Other threads in the warp wait for the insertion to complete

### Performance Optimizations

1. **Warp-Aligned Design**
   - Bucket size matches GPU warp size (64 threads)
   - Enables efficient warp-level operations
   - Reduces thread divergence

2. **Memory Access Optimization**
   - Single-thread atomic operations minimize memory contention
   - Cooperative groups enable efficient thread communication
   - Linear probing is cache-friendly

3. **Load Balancing**
   - Configurable load factor (default: 0.5)
   - Allows tuning based on expected data distribution
   - Helps balance memory usage and collision probability

## Kernel Walkthrough

The insertion kernel (`insert_into_hashtable`) is the core of the hash set implementation. Here's a detailed breakdown of how it works:

### Kernel Parameters
```cpp
template <typename key_type, int bucket_size>
__global__ void insert_into_hashtable(
    const key_type* keys,           // Input keys to insert
    key_type* hash_table,          // Hash table storage
    const key_type sentinel_key,   // Value marking empty slots
    const std::size_t num_keys,    // Number of keys to insert
    const std::size_t num_buckets  // Total number of buckets
)
```

### Thread Organization
```cpp
const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
auto block = cooperative_groups::this_thread_block();
auto tile = cooperative_groups::tiled_partition<bucket_size>(block);
```
- Each thread gets a unique ID
- Threads are organized into cooperative groups
- Threads are partitioned into tiles matching the bucket size (64 threads)

### Key Loading
```cpp
bool do_insert = false;
key_type key_to_insert{};

if (thread_id < num_keys) {
    key_to_insert = keys[thread_id];
    do_insert = true;
}
```
- Each thread loads a key from the input array
- Threads beyond `num_keys` are marked as inactive

### Hash Computation
```cpp
uint32_t hash_x = 0x9e3779b9;
uint32_t hash_y = 0x9e7019b9;
const uint32_t prime_divisor = 1 << 31;
const auto hash = (((hash_x ^ key_to_insert) + hash_y) % prime_divisor);
auto bucket_index = hash % num_buckets;
```
- Computes a hash value for the key
- Maps the hash to a bucket index

### Insertion Process
```cpp
auto work_queue = __ballot64(do_insert);
while (work_queue) {
    auto cur_rank = __ffsll(work_queue) - 1;
    auto cur_key = tile.shfl(key_to_insert, cur_rank);
    auto lane_id = tile.thread_rank();

    if (cur_rank == lane_id) {
        bool success = false;
        auto offset_wihin_bucket = 0;
        while (!success) {
            success = __hip_atomic_compare_exchange_strong(
                &hash_table[bucket_offset + offset_wihin_bucket],
                &expected,
                desired,
                __ATOMIC_RELAXED,
                __ATOMIC_RELAXED,
                __HIP_MEMORY_SCOPE_SYSTEM);
            // ... linear probing logic ...
        }
    }
}
```
1. **Work Queue Management**
   - Uses `__ballot64` to create a bitmask of threads with keys to insert
   - Processes one thread's insertion at a time

2. **Thread Selection**
   - `__ffsll` finds the next thread with work to do
   - The selected thread performs the insertion
   - Other threads in the warp wait

3. **Atomic Insertion**
   - Uses atomic compare-and-exchange to insert the key
   - Ensures thread-safe insertion

4. **Linear Probing**
   - If the bucket is full, moves to the next bucket
   - Continues until finding an empty slot or checking all buckets
   - Wraps around to the beginning if needed

### Key Optimizations

1. **Warp-Level Coordination**
   - Uses cooperative groups for efficient thread communication
   - Threads within a warp work together on insertions
   - Minimizes thread divergence

2. **Memory Access Patterns**
   - Single-thread atomic operations reduce contention
   - Linear probing provides good cache locality
   - Keeps data local to the loading thread

3. **Load Factor**
   - Configurable load factor prevents excessive collisions
   - Linear probing distributes keys across buckets
   - Warp-level work queue ensures fair thread utilization

## Usage

```cpp
// Example initialization
const std::size_t num_keys = 1'000'000;
const float load_factor = 0.5f;
const int bucket_size = 64;  // Must match GPU warp size

// Calculate number of buckets based on load factor
const std::size_t num_buckets =
    std::max(std::size_t{1},
             static_cast<std::size_t>(num_keys / bucket_size / load_factor));

// Initialize hash set with sentinel values
thrust::device_vector<key_type> d_hash_table(capacity, sentinel_key);

// Launch kernel for parallel insertion
insert_into_hashtable<key_type, bucket_size><<<num_blocks, bucket_size>>>(
    d_keys.data().get(), d_hash_table.data().get(), sentinel_key, num_keys,
    num_buckets);
```

## Performance Considerations

1. **Load Factor**
   - Default load factor of 0.5 provides good balance between space and performance
   - Higher load factors increase collision probability
   - Lower load factors waste memory but reduce collisions

2. **Memory Access Patterns**
   - Linear probing provides good cache locality
   - Single-thread atomic operations reduce memory contention
   - Warp-aligned design enables efficient memory access

3. **Thread Utilization**
   - Warp-level operations maximize thread efficiency
   - Cooperative groups enable efficient thread communication
   - Single-thread insertion reduces unnecessary memory operations

## Future Extensions

This implementation focuses on parallel insertion. Future versions could include:
- Parallel lookup operations
- Delete operations
- Dynamic resizing
- Different collision resolution strategies
- Support for different key types and hash functions


