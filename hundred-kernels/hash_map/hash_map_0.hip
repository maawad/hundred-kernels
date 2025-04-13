#include <hip/hip_runtime.h>
#include <algorithm>
#include <random>

#include <hip/hip_cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <typename key_type, typename value_type, int bucket_size>
__global__ void insert_into_hashtable(const key_type* keys,
                                      const value_type* values,
                                      key_type* hash_table_keys,
                                      value_type* hash_table_values,
                                      const key_type sentinel_key,
                                      const std::size_t num_keys,
                                      const std::size_t num_buckets) {
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block = cooperative_groups::this_thread_block();
  auto tile = cooperative_groups::tiled_partition<bucket_size>(block);

  if ((thread_id - tile.thread_rank()) >= num_keys) {
    return;
  }

  bool do_insert = false;
  key_type key_to_insert{};
  value_type value_to_insert{};
  // load the input
  if (thread_id < num_keys) {
    key_to_insert = keys[thread_id];
    value_to_insert = values[thread_id];
    do_insert = true;
  }

  uint32_t hash_x = 0x9e3779b9;
  uint32_t hash_y = 0x9e7019b9;
  const uint32_t prime_divisor = 1 << 31;
  const auto hash = (((hash_x ^ key_to_insert) + hash_y) % prime_divisor);

  auto bucket_index = hash % num_buckets;
  auto bucket_offset = bucket_index * bucket_size;
  const auto initial_bucket_index = bucket_index;

  // Do the insertion
  auto work_queue = __ballot64(do_insert);
  while (work_queue) {
    auto cur_rank = __ffsll(work_queue) - 1;
    auto cur_key = tile.shfl(key_to_insert, cur_rank);
    auto lane_id = tile.thread_rank();

    auto expected = sentinel_key;
    auto desired = key_to_insert;

    bool success = false;
    auto offset_wihin_bucket = 0;
    if (cur_rank == lane_id) {
      while (!success) {
        success = __hip_atomic_compare_exchange_strong(
            &hash_table_keys[bucket_offset + offset_wihin_bucket], &expected,
            desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
            __HIP_MEMORY_SCOPE_SYSTEM);
        if (success) {
          hash_table_values[bucket_offset + offset_wihin_bucket] =
              value_to_insert;
        } else {
          offset_wihin_bucket++;
          if (offset_wihin_bucket == bucket_size) {
            bucket_index = (bucket_index + 1) % num_buckets;
            bucket_offset = bucket_index * bucket_size;
            if (bucket_index == initial_bucket_index && !success) {
              break;
            }
          }
        }
      }
    }

    if (tile.thread_rank() == cur_rank) {
      do_insert = false;
    }
    work_queue = __ballot64(do_insert);
  }
}

int main() {
  const std::size_t num_keys = 1'000'000;

  const float load_factor = 0.5f;
  const int bucket_size = 64;

  const std::size_t num_buckets =
      std::max(std::size_t{1},
               static_cast<std::size_t>(num_keys / bucket_size / load_factor));
  const std::size_t capacity = num_buckets * bucket_size;

  using key_type = int;
  using value_type = int;
  const auto sentinel_key = std::numeric_limits<key_type>::max();
  const auto sentinel_value = std::numeric_limits<value_type>::max();

  thrust::device_vector<key_type> d_hash_table_keys(capacity, sentinel_key);
  thrust::device_vector<value_type> d_hash_table_values(capacity,
                                                        sentinel_value);

  thrust::device_vector<key_type> d_keys(num_keys);
  thrust::device_vector<value_type> d_values(num_keys);
  thrust::host_vector<key_type> h_keys(num_keys);
  thrust::host_vector<value_type> h_values(num_keys);

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<key_type> dist(
      0, std::numeric_limits<key_type>::max() - 1);

  std::generate(h_keys.begin(), h_keys.end(), [&]() { return dist(rng); });
  std::generate(h_values.begin(), h_values.end(), [&]() { return dist(rng); });

  d_keys = h_keys;
  d_values = h_values;

  const auto block_size = 1024;
  const auto num_blocks = (num_keys + bucket_size - 1) / bucket_size;

  insert_into_hashtable<key_type, value_type, bucket_size>
      <<<num_blocks, bucket_size>>>(d_keys.data().get(), d_values.data().get(),
                                    d_hash_table_keys.data().get(),
                                    d_hash_table_values.data().get(),
                                    sentinel_key, num_keys, num_buckets);

  const auto err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    printf("Error: %s\n", hipGetErrorString(err));
    return 1;
  }

  return 0;
}
