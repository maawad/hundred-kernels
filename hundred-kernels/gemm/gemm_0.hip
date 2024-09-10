#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>

#include <iostream>

__global__ void gemm(const double* A, const double* B, double* C, std::size_t M,
                     std::size_t N, std::size_t K, double alpha, double beta) {
  const std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    double value = 0.0;
    for (std::size_t i = 0; i < K; ++i) {
      value += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * value + beta * C[row * N + col];
  }
}

int main() {
  const std::size_t M = 1024;
  const std::size_t N = 1024;
  const std::size_t K = 1024;

  const double alpha = 1.0;
  const double beta = 0.0;

  thrust::device_vector<double> A(M * K, 1.0);
  thrust::device_vector<double> B(K * N, 1.0);
  thrust::device_vector<double> C(M * N, 0.0);

  const std::size_t block_size = 16;
  const std::size_t grid_x = (N + block_size - 1) / block_size;
  const std::size_t grid_y = (M + block_size - 1) / block_size;

  dim3 block(block_size, block_size);
  dim3 grid(grid_x, grid_y);

  gemm<<<grid, block>>>(A.data().get(), B.data().get(), C.data().get(), M, N, K,
                        alpha, beta);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    std::cout << "Kernel failed with error: " << hipGetErrorString(status)
              << "\n";
    return -1;
  } else {
    std::cout << "Success!\n";
  }

  return 0;
}
