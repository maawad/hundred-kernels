#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include <iostream>

__global__ void jacobi_iteration(const double* A, const double* b,
                                 const double* x_old, double* x_new,
                                 std::size_t N) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double sum = 0.0;
    for (std::size_t j = 0; j < N; ++j) {
      if (i != j) {
        sum += A[i * N + j] * x_old[j];
      }
    }
    x_new[i] = (b[i] - sum) / A[i * N + i];
  }
}

int main() {
  const std::size_t N = 1024;
  const std::size_t max_iterations = 1;
  const double tolerance = 1e-6;

  thrust::device_vector<double> A(N * N, 1.0);
  thrust::device_vector<double> b(N, 1.0);
  thrust::device_vector<double> x_old(N, 0.0);
  thrust::device_vector<double> x_new(N, 0.0);

  for (std::size_t i = 0; i < N; ++i) {
    A[i * N + i] = N + 1.0;
  }

  const std::size_t block_size = 256;
  const std::size_t grid_size = (N + block_size - 1) / block_size;

  for (std::size_t iter = 0; iter < max_iterations; ++iter) {
    jacobi_iteration<<<grid_size, block_size>>>(A.data().get(), b.data().get(),
                                                x_old.data().get(),
                                                x_new.data().get(), N);

    const auto status = hipDeviceSynchronize();

    if (status != hipSuccess) {
      std::cout << "Kernel failed.\n";
      return -1;
    }

    double diff =
        thrust::inner_product(x_new.begin(), x_new.end(), x_old.begin(), 0.0,
                              thrust::plus<double>(), thrust::minus<double>());

    if (diff < tolerance) {
      std::cout << "Converged after " << iter + 1
                << " iterations with diff: " << diff << std::endl;
      break;
    }

    thrust::copy(x_new.begin(), x_new.end(), x_old.begin());
  }

  return 0;
}
