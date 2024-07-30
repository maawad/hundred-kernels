#include <thrust/device_vector.h>

#include <cstdint>
#include <iostream>

__global__ void inclusive_scan(double* v_a, double* result, std::size_t count) {
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id < count) {
    double sum{0};
    for (std::size_t prev = 0; prev <= thread_id; prev++) {
      sum += v_a[prev];
    }
    result[thread_id] = sum;
  }
}

int main() {
  const std::size_t count{1'000'000};
  thrust::device_vector<double> data(count, 1);
  thrust::device_vector<double> result(count);

  const std::size_t block_size{512};
  const std::size_t num_blocks{(count + block_size - 1) / block_size};

  inclusive_scan<<<num_blocks, block_size>>>(data.data().get(),
                                             result.data().get(), count);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess || result[0] != 1 || result[count - 1] != (count)) {
    std::cout << "Kernel failed.\n";
    std::cout << result[0] << " \n";
    std::cout << result[count - 1] << " \n";

    return -1;
  } else {
    std::cout << "Success!\n";
  }
}