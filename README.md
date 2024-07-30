# Modern HIP examples

This repository contains a collection of modern HIP GPU kernels. The kernel examples are stored in a per-problem directory, each directory containing files numbered starting from 0, with lower numbers indicating simple kernels and higher numbers indicating more optimized kernels.



# Build
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


# Run
To run all examples,
```shell
./run_all 
```
Which should print:
```shell
Build Directory: build 

build/add_0 --------------------------------------- | SUCCESS
build/add_1 --------------------------------------- | SUCCESS
build/exclusive_scan_0 ---------------------------- | SUCCESS
build/inclusive_scan_0 ---------------------------- | SUCCESS
build/multiply_0 ---------------------------------- | SUCCESS
build/multiply_1 ---------------------------------- | SUCCESS
build/multiply_add_0 ------------------------------ | SUCCESS
build/multiply_add_1 ------------------------------ | SUCCESS
build/reduction_0 --------------------------------- | SUCCESS
build/reduction_1 --------------------------------- | SUCCESS
build/reduction_2 --------------------------------- | SUCCESS
build/reduction_3 --------------------------------- | SUCCESS

Successful tests: 12
Failed tests: 0
```

To run the reduction example,
```shell
./build/reduction_3
```
Which on success, will print:
```shell
Success!
```


