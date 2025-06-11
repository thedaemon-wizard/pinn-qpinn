#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "nsga2_optimizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nsga2_optimizer, m) {
    m.doc() = "NSGA-II optimizer for multi-objective optimization with SOLID principles";
    
    // Individual class binding
    py::class_<nsga2::Individual, std::shared_ptr<nsga2::Individual>>(m, "Individual")
        .def(py::init<size_t>())
        .def(py::init<const std::vector<double>&>())
        .def_readwrite("parameters", &nsga2::Individual::parameters)
        .def_readwrite("objectives", &nsga2::Individual::objectives)
        .def_readwrite("rank", &nsga2::Individual::rank)
        .def_readwrite("crowding_distance", &nsga2::Individual::crowding_distance)
        .def("dominates", &nsga2::Individual::dominates)
        .def("reset", &nsga2::Individual::reset);
    
    // Population class binding
    py::class_<nsga2::Population, std::shared_ptr<nsga2::Population>>(m, "Population")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def("add", &nsga2::Population::add)
        .def("clear", &nsga2::Population::clear)
        .def("size", &nsga2::Population::size)
        .def("__len__", &nsga2::Population::size)
        .def("__getitem__", [](const nsga2::Population& pop, size_t idx) {
            if (idx >= pop.size()) {
                throw py::index_error();
            }
            return pop[idx];
        })
        .def("__iter__", [](const nsga2::Population& pop) {
            return py::make_iterator(pop.begin(), pop.end());
        }, py::keep_alive<0, 1>());
    
    // Abstract crowding distance calculator interface
    py::class_<nsga2::ICrowdingDistanceCalculator, std::shared_ptr<nsga2::ICrowdingDistanceCalculator>>(m, "ICrowdingDistanceCalculator")
        .def("calculate", &nsga2::ICrowdingDistanceCalculator::calculate)
        .def("getName", &nsga2::ICrowdingDistanceCalculator::getName);
    
    // Traditional crowding distance
    py::class_<nsga2::TraditionalCrowdingDistance, nsga2::ICrowdingDistanceCalculator, 
               std::shared_ptr<nsga2::TraditionalCrowdingDistance>>(m, "TraditionalCrowdingDistance")
        .def(py::init<>());
    
    // Equidistant selection crowding distance
    py::class_<nsga2::EquidistantSelectionCrowdingDistance, nsga2::ICrowdingDistanceCalculator,
               std::shared_ptr<nsga2::EquidistantSelectionCrowdingDistance>>(m, "EquidistantSelectionCrowdingDistance")
        .def(py::init<size_t>(), py::arg("selection_size") = 0)
        .def("setSelectionSize", &nsga2::EquidistantSelectionCrowdingDistance::setSelectionSize);
    
    // CrowdingDistanceFactory
    py::class_<nsga2::CrowdingDistanceFactory>(m, "CrowdingDistanceFactory")
        .def_static("create", &nsga2::CrowdingDistanceFactory::create,
                    py::arg("type"), py::arg("selection_size") = 0);
    
    // CrowdingDistanceFactory::Type enum
    py::enum_<nsga2::CrowdingDistanceFactory::Type>(m, "CrowdingDistanceType")
        .value("Traditional", nsga2::CrowdingDistanceFactory::Type::Traditional)
        .value("EquidistantSelection", nsga2::CrowdingDistanceFactory::Type::EquidistantSelection);
    
    // NSGA2Config
    py::class_<nsga2::NSGA2Config>(m, "NSGA2Config")
        .def(py::init<>())
        .def_readwrite("population_size", &nsga2::NSGA2Config::population_size)
        .def_readwrite("max_generations", &nsga2::NSGA2Config::max_generations)
        .def_readwrite("n_objectives", &nsga2::NSGA2Config::n_objectives)
        .def_readwrite("lower_bounds", &nsga2::NSGA2Config::lower_bounds)
        .def_readwrite("upper_bounds", &nsga2::NSGA2Config::upper_bounds)
        .def_readwrite("rex_xi", &nsga2::NSGA2Config::rex_xi)
        .def_readwrite("n_parents", &nsga2::NSGA2Config::n_parents)
        .def_readwrite("n_children", &nsga2::NSGA2Config::n_children)
        .def_readwrite("random_seed", &nsga2::NSGA2Config::random_seed)
        .def_readwrite("verbose", &nsga2::NSGA2Config::verbose)
        .def_readwrite("progress_interval", &nsga2::NSGA2Config::progress_interval)
        .def_readwrite("crowding_type", &nsga2::NSGA2Config::crowding_type);
    
    // NSGA2Optimizer
    py::class_<nsga2::NSGA2Optimizer>(m, "NSGA2Optimizer")
        .def(py::init<const nsga2::NSGA2Config&>())
        
        // Strategy setters
        .def("setCrowdingDistanceCalculator", 
             py::overload_cast<std::shared_ptr<nsga2::ICrowdingDistanceCalculator>>(
                 &nsga2::NSGA2Optimizer::setCrowdingDistanceCalculator),
             "Set custom crowding distance calculator")
        
        // Main optimize method
        .def("optimize", [](nsga2::NSGA2Optimizer& self,
                           const std::vector<nsga2::ObjectiveFunction>& objectives,
                           py::object callback,
                           py::object batch_evaluator) {
            
            // Wrap Python callback
            std::function<void(size_t, const nsga2::Population&)> cpp_callback = nullptr;
            
            if (!callback.is_none()) {
                cpp_callback = [callback](size_t generation, const nsga2::Population& pop) {
                    py::list pop_list;
                    for (size_t i = 0; i < pop.size(); ++i) {
                        py::dict ind_dict;
                        ind_dict["parameters"] = pop[i]->parameters;
                        ind_dict["objectives"] = pop[i]->objectives;
                        ind_dict["rank"] = pop[i]->rank;
                        ind_dict["crowding_distance"] = pop[i]->crowding_distance;
                        pop_list.append(ind_dict);
                    }
                    
                    py::gil_scoped_acquire acquire;
                    callback(generation, pop_list);
                };
            }
            
            // Wrap batch evaluator
            std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)> cpp_batch_evaluator = nullptr;
            
            if (!batch_evaluator.is_none()) {
                cpp_batch_evaluator = [batch_evaluator](const std::vector<std::vector<double>>& params_batch) {
                    py::gil_scoped_acquire acquire;
                    
                    py::list py_params_batch;
                    for (const auto& params : params_batch) {
                        py_params_batch.append(params);
                    }
                    
                    py::object result = batch_evaluator(py_params_batch);
                    
                    std::vector<std::vector<double>> objectives_batch;
                    for (auto item : result) {
                        std::vector<double> obj_vec;
                        for (auto obj : item) {
                            obj_vec.push_back(obj.cast<double>());
                        }
                        objectives_batch.push_back(obj_vec);
                    }
                    
                    return objectives_batch;
                };
            }
            
            return self.optimize(objectives, cpp_callback, cpp_batch_evaluator);
        },
        py::arg("objectives"),
        py::arg("callback") = py::none(),
        py::arg("batch_evaluator") = py::none(),
        "Run NSGA-II optimization with optional batch evaluation")
        
        // Pareto front getter
        .def("get_pareto_front", [](const nsga2::NSGA2Optimizer& self) {
            auto pareto_front = self.get_pareto_front();
            py::list result;
            for (const auto& ind : pareto_front) {
                py::dict ind_dict;
                ind_dict["parameters"] = ind->parameters;
                ind_dict["objectives"] = ind->objectives;
                ind_dict["rank"] = ind->rank;
                ind_dict["crowding_distance"] = ind->crowding_distance;
                result.append(ind_dict);
            }
            return result;
        })
        
        // History accessors
        .def("get_fitness_history", &nsga2::NSGA2Optimizer::get_fitness_history)
        .def("get_mean_fitness_history", &nsga2::NSGA2Optimizer::get_mean_fitness_history)
        .def("get_best_fitness", &nsga2::NSGA2Optimizer::get_best_fitness)
        .def("get_current_generation", &nsga2::NSGA2Optimizer::get_current_generation);
    
    // Helper function to create equidistant selection crowding distance easily
    m.def("create_equidistant_crowding", 
          [](size_t selection_size) {
              return std::make_shared<nsga2::EquidistantSelectionCrowdingDistance>(selection_size);
          },
          py::arg("selection_size") = 0,
          "Create an equidistant selection crowding distance calculator");
}