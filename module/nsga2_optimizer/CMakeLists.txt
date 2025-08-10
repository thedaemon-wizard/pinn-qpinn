cmake_minimum_required(VERSION 3.12)
project(nsga2_optimizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create the module
pybind11_add_module(nsga2_optimizer
    src/nsga2_optimizer.cpp
    src/python_bindings.cpp
)

# Set properties
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nsga2_optimizer PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_features(nsga2_optimizer PRIVATE cxx_std_17)
set_target_properties(nsga2_optimizer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Add optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nsga2_optimizer PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(nsga2_optimizer PRIVATE /O2)
endif()