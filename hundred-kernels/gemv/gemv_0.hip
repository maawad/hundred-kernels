#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

template <typename T>
__global__ void gemv(const T* A, const T* x, T* y, int m, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m) {
    T sum = 0;
    for (int i = 0; i < n; i++) {
      sum += A[row * n + i] * x[i];
    }
    y[row] = sum;
  }
}

int main() {
  int m = 1024;
  int n = 1024;

  thrust::device_vector<float> A(m * n);
  thrust::device_vector<float> x(n);
  thrust::device_vector<float> y(m);

  constexpr int block_size = 1024;
  const int num_blocks = (m + block_size - 1) / block_size;

  gemv<<<num_blocks, block_size>>>(A.data().get(), x.data().get(),
                                  y.data().get(), m, n);

  const auto sucess = hipDeviceSynchronize();
  if (sucess != hipSuccess) {
    std::cerr << "Kernel launch failed" << std::endl;
    return 1;
  }

  thrust::host_vector<float> h_y = y;
  for (int i = 0; i < std::min(m, 10); i++) {
    std::cout << h_y[i] << " ";
  }
}
