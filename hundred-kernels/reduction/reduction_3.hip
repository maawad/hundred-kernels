#include <thrust/device_vector.h>

#include <cstdint>
#include <iostream>
#include <rocprim/block/block_reduce.hpp>

template <std::size_t block_size>
__global__ void reduction(double* input, double* result, std::size_t count) {
  using BlockReduce =
      rocprim::block_reduce<double, block_size,
                            rocprim::block_reduce_algorithm::using_warp_reduce,
                            1>;
  __shared__ typename BlockReduce::storage_type temp_storage;

  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const auto stride = blockDim.x * gridDim.x;
  const auto count_cieled =
      ((count + block_size - 1) / block_size) * block_size;
  for (auto id = thread_id; id < count_cieled; id += stride) {
    const auto value = id < count ? input[thread_id] : 0;
    double block_value{0};
    BlockReduce().reduce(value, block_value, temp_storage);
    if (threadIdx.x == 0) {
      atomicAdd(result, block_value);
    }
  }
}

int main() {
  const std::size_t count{1'000'000};
  thrust::device_vector<double> input(count, 1);
  thrust::device_vector<double> result(1, 0);

  constexpr std::size_t block_size{512};
  const std::size_t num_blocks{1024};

  reduction<block_size><<<num_blocks, block_size>>>(input.data().get(),
                                                    result.data().get(), count);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess || result[0] != count) {
    std::cout << "Kernel failed.\n";
    return -1;
  } else {
    std::cout << "Success!\n";
  }
}