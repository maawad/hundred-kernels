# Hundred Kernels

A collection of high-performance GPU kernels implemented using HIP (AMD's GPU programming platform). This project demonstrates various parallel algorithms and data structures optimized for GPU execution, showcasing different GPU programming patterns and optimization techniques.

## Versioning Pattern

Each algorithm is implemented in multiple versions, with increasing optimization levels:
- Files ending with `_0` represent the basic implementation
- Higher numbered versions (e.g., `_1`, `_2`, `_3`) show progressively more optimized implementations
- Each version demonstrates different GPU programming patterns and optimization techniques

## Examples

The project includes the following examples, all successfully building and running:

### Basic Operations
- `add_0`, `add_1`: Vector addition implementations with increasing optimization
- `multiply_0`, `multiply_1`: Vector multiplication implementations with increasing optimization
- `multiply_add_0`, `multiply_add_1`: Fused multiply-add operations with increasing optimization

### Linear Algebra
- `gemm_0`: General matrix-matrix multiplication (basic implementation)
- `gemv_0`, `gemv_1`: General matrix-vector multiplication with increasing optimization

### Data Structures
- `hash_map_0`: GPU hash map implementation (basic version)
- `hash_set_0`: GPU hash set implementation (basic version, see [hash_set/README.md](hash_set/README.md) for details)
- `histogram_0`: Parallel histogram computation (basic version)

### Parallel Algorithms
- `exclusive_scan_0`: Exclusive prefix sum (basic implementation)
- `inclusive_scan_0`: Inclusive prefix sum (basic implementation)
- `reduction_0`, `reduction_1`, `reduction_2`, `reduction_3`: Reduction implementations with increasing optimization
- `jacobi_0`: Jacobi iteration for solving linear systems (basic implementation)

## Building

Use the [rebuild](./rebuild) script to quickly rebuild the CMake-based project. The script has the following options:
```shell
Usage: ./rebuild [options]
Options:
  -a, --arch          Set HIP architecture (default: gfx1100)
  -d, --dir           Set build directory (default: build)
  -t, --type          Set build type (default: Release)
  -T, --target        Set build target (default: all)
  -h, --help          Show this help message
```

For example to compile for MI210, you can execute:
```shell
./rebuild --arch gfx90a
```

## Running

To run all examples:
```shell
./run_all
```

Which should print:
```shell
Build Directory: build

build/add_0 --------------------------------------- | SUCCESS
build/add_1 --------------------------------------- | SUCCESS
build/exclusive_scan_0 ---------------------------- | SUCCESS
build/gemm_0 -------------------------------------- | SUCCESS
build/gemv_0 -------------------------------------- | SUCCESS
build/gemv_1 -------------------------------------- | SUCCESS
build/hash_map_0 ---------------------------------- | SUCCESS
build/hash_set_0 ---------------------------------- | SUCCESS
build/histogram_0 --------------------------------- | SUCCESS
build/inclusive_scan_0 ---------------------------- | SUCCESS
build/jacobi_0 ------------------------------------ | SUCCESS
build/multiply_0 ---------------------------------- | SUCCESS
build/multiply_1 ---------------------------------- | SUCCESS
build/multiply_add_0 ------------------------------ | SUCCESS
build/multiply_add_1 ------------------------------ | SUCCESS
build/reduction_0 --------------------------------- | SUCCESS
build/reduction_1 --------------------------------- | SUCCESS
build/reduction_2 --------------------------------- | SUCCESS
build/reduction_3 --------------------------------- | SUCCESS

Successful tests: 19
Failed tests: 0
```

To run a specific example, for instance the reduction example:
```shell
./build/reduction_3
```

Which on success, will print:
```shell
Success!
```

## Project Structure

Each example is organized in its own directory under the main project root:
```
hundred-kernels/
├── add/
├── exclusive_scan/
├── gemm/
├── gemv/
├── hash_map/
├── hash_set/
├── histogram/
├── inclusive_scan/
├── jacobi/
├── multiply/
├── multiply_add/
└── reduction/
```

## Documentation

Detailed documentation for each example can be found in their respective directories. For example:
- [Hash Set Implementation](hundred-kernels/hash_set/README.md)
