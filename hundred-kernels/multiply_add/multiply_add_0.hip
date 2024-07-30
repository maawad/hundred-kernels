#include <thrust/device_vector.h>

#include <cstdint>
#include <iostream>

__global__ void multiply_add(double* v_a, double* v_b, double* v_c,
                             double* result, std::size_t count) {
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id < count) {
    result[thread_id] = v_a[thread_id] * v_b[thread_id] + v_c[thread_id];
  }
}

int main() {
  const std::size_t count{1'000'000};
  thrust::device_vector<double> v_a(count, 2);
  thrust::device_vector<double> v_b(count, 3);
  thrust::device_vector<double> v_c(count, 4);
  thrust::device_vector<double> result(count);

  const std::size_t block_size{512};
  const std::size_t num_blocks{(count + block_size - 1) / block_size};

  multiply_add<<<num_blocks, block_size>>>(v_a.data().get(), v_b.data().get(),
                                           v_c.data().get(),
                                           result.data().get(), count);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess || result[0] != 10 || result[count - 1] != 10) {
    std::cout << "Kernel failed.\n";
    return -1;
  } else {
    std::cout << "Success!\n";
  }
}