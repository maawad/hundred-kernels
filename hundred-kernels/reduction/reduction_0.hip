#include <thrust/device_vector.h>

#include <cstdint>
#include <iostream>

__global__ void reduction(double* input, double* result, std::size_t count) {
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id < count) {
    const auto value = input[thread_id];
    atomicAdd(result, value);
  }
}

int main() {
  const std::size_t count{1'000'000};
  thrust::device_vector<double> input(count, 1);
  thrust::device_vector<double> result(1, 0);

  const std::size_t block_size{512};
  const std::size_t num_blocks{(count + block_size - 1) / block_size};

  reduction<<<num_blocks, block_size>>>(input.data().get(), result.data().get(),
                                        count);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess || result[0] != count) {
    std::cout << "Kernel failed.\n";
    return -1;
  } else {
    std::cout << "Success!\n";
  }
}