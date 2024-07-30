#include <thrust/device_vector.h>

#include <cstdint>
#include <iostream>

__global__ void multiply(double* v_a, double* v_b, double* result,
                         std::size_t count) {
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto id = thread_id; id < count; id += stride) {
    result[id] = v_a[id] * v_b[id];
  }
}

int main() {
  const std::size_t count{1'000'000};
  thrust::device_vector<double> v_a(count, 1);
  thrust::device_vector<double> v_b(count, 3);
  thrust::device_vector<double> result(count);

  const std::size_t block_size{512};
  const std::size_t num_blocks{1024};

  multiply<<<num_blocks, block_size>>>(v_a.data().get(), v_b.data().get(),
                                       result.data().get(), count);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess || result[0] != 3 || result[count - 1] != 3) {
    std::cout << "Kernel failed.\n";
    return -1;
  } else {
    std::cout << "Success!\n";
  }
}