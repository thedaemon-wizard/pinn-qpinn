#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "rcga_optimizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rcga_optimizer, m) {
    m.doc() = "RCGA optimizer for quantum circuit optimization";
    
    // RCGAConfig
    py::class_<rcga::RCGAConfig>(m, "RCGAConfig")
        .def(py::init<>())
        .def_readwrite("population_size", &rcga::RCGAConfig::population_size)
        .def_readwrite("max_generations", &rcga::RCGAConfig::max_generations)
        .def_readwrite("num_parents", &rcga::RCGAConfig::num_parents)
        .def_readwrite("num_children", &rcga::RCGAConfig::num_children)
        .def_readwrite("xi", &rcga::RCGAConfig::xi)
        .def_readwrite("min_val", &rcga::RCGAConfig::min_val)
        .def_readwrite("max_val", &rcga::RCGAConfig::max_val)
        .def_readwrite("random_seed", &rcga::RCGAConfig::random_seed)
        .def_readwrite("verbose", &rcga::RCGAConfig::verbose)
        .def_readwrite("use_lhs", &rcga::RCGAConfig::use_lhs)
        .def_readwrite("progress_interval", &rcga::RCGAConfig::progress_interval);
    
    // RCGAOptimizer
    py::class_<rcga::RCGAOptimizer>(m, "RCGAOptimizer")
        .def(py::init<const rcga::RCGAConfig&>(), py::arg("config") = rcga::RCGAConfig())
        .def("optimize", 
            [](rcga::RCGAOptimizer& self, 
               size_t dimension,
               std::function<double(const std::vector<double>&)> eval_func,
               py::object batch_eval_func_py,
               py::object progress_callback_py) {
                
                // Create evaluator
                std::shared_ptr<rcga::FitnessEvaluator> evaluator;
                
                // Check if batch_eval_func is provided
                if (!batch_eval_func_py.is_none()) {
                    auto batch_eval_func = batch_eval_func_py.cast<std::function<std::vector<double>(const std::vector<std::vector<double>>&)>>();
                    evaluator = std::make_shared<rcga::PythonFitnessEvaluator>(eval_func, batch_eval_func);
                } else {
                    evaluator = std::make_shared<rcga::PythonFitnessEvaluator>(eval_func);
                }
                
                // Check if progress_callback is provided
                if (!progress_callback_py.is_none()) {
                    auto progress_callback = progress_callback_py.cast<rcga::ProgressCallback>();
                    return self.optimize(dimension, evaluator, progress_callback);
                } else {
                    return self.optimize(dimension, evaluator);
                }
            }, 
            py::arg("dimension"), 
            py::arg("eval_func"), 
            py::arg("batch_eval_func") = py::none(),
            py::arg("progress_callback") = py::none(),
            "Optimize using RCGA algorithm")
        .def("get_best_solution", &rcga::RCGAOptimizer::getBestSolution)
        .def("get_best_fitness", &rcga::RCGAOptimizer::getBestFitness)
        .def("get_fitness_history", &rcga::RCGAOptimizer::getFitnessHistory)
        .def("get_mean_fitness_history", &rcga::RCGAOptimizer::getMeanFitnessHistory)
        .def("get_current_generation", &rcga::RCGAOptimizer::getCurrentGeneration)
        .def("set_config", &rcga::RCGAOptimizer::setConfig)
        .def("get_config", &rcga::RCGAOptimizer::getConfig);
}