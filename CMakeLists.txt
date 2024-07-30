cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(bght LANGUAGES CXX HIP)


file(GLOB_RECURSE example_targets "${CMAKE_SOURCE_DIR}/hundred-kernels/*.hip")

function(add_example target_name source_file)
    message(STATUS "Adding target: ${target_name}")
    add_executable(${target_name} ${source_file})
endfunction()


foreach(example ${example_targets})
    get_filename_component(target ${example} NAME_WE)
    add_example(${target} ${example})
endforeach()