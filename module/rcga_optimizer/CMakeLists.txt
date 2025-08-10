cmake_minimum_required(VERSION 3.12)
project(rcga_optimizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/rcga_optimizer.cpp
    src/rex_crossover.cpp
    src/jgg_selection.cpp
    src/python_bindings.cpp
)

# Create Python module
pybind11_add_module(rcga_optimizer ${SOURCES})

# Set properties
target_compile_features(rcga_optimizer PRIVATE cxx_std_17)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rcga_optimizer PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(rcga_optimizer PRIVATE /O2)
endif()