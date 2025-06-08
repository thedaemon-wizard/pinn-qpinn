from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

ext_modules = [
    Pybind11Extension(
        "rcga_optimizer",
        [
            "src/rcga_optimizer.cpp",
            "src/rex_crossover.cpp", 
            "src/jgg_selection.cpp",
            "src/python_bindings.cpp"
        ],
        include_dirs=["include"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"]
    ),
]

setup(
    name="rcga_optimizer",
    version='1.0.0',
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)