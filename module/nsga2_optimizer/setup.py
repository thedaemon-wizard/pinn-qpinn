from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "nsga2_optimizer",
        ["src/nsga2_optimizer.cpp", "src/python_bindings.cpp"],
        include_dirs=["include",
            # pybind11のインクルードディレクトリ
            pybind11.get_include(),],
        cxx_std=17,  # C++17標準を使用
        define_macros=[('USE_OPENMP', '1')],
        extra_compile_args=['-O3', '-fopenmp', '-march=native'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name="nsga2_optimizer",
    version="2.0.0",
    description="NSGA-II optimizer with equidistant selection and SOLID principles",
    long_description="""
    A high-performance NSGA-II (Non-dominated Sorting Genetic Algorithm II) 
    implementation in C++ with Python bindings. Features include:
    - Traditional and equidistant selection crowding distance methods
    - SOLID principles-based architecture
    - OpenMP parallelization
    - Batch evaluation support
    - REX crossover operator
    - Latin Hypercube Sampling initialization
    """,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.12",
    install_requires=[
        "numpy>=2.2.0",
        "pybind11>=2.6.0",
    ],
)