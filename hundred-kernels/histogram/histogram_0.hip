#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <iostream>

__global__ void histogram(double* data, double* bins,
                          const std::size_t num_elements,
                          const std::size_t num_bins, const double bin_size) {
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id < num_elements) {
    std::size_t bin_index =
        static_cast<std::size_t>(data[thread_id] / bin_size);
    bin_index = min(bin_index, num_bins - 1);

    atomicAdd(&bins[bin_index], 1);
  }
}

int main() {
  const std::size_t num_elements{1'000'000};
  const std::size_t num_bins{1024};
  const double bin_size{32.};

  thrust::device_vector<double> data(num_elements, 1);
  thrust::device_vector<double> bins(num_bins, 0);

  thrust::sequence(data.begin(), data.end());

  const std::size_t block_size{512};
  const std::size_t num_blocks{(num_elements + block_size - 1) / block_size};

  histogram<<<num_blocks, block_size>>>(data.data().get(), bins.data().get(),
                                        num_elements, num_bins, bin_size);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    std::cout << "Kernel failed with error: " << hipGetErrorString(status)
              << "\n";
    return -1;
  } else {
    std::cout << "Success!\n";
  }
}
