NSGA-II Optimizer
=================

This chapter documents the high-performance C++ implementation of the NSGA-II (Non-dominated Sorting Genetic Algorithm II) optimizer with Python bindings.

Overview
--------

The NSGA-II optimizer is implemented in C++ for performance and provides Python bindings via pybind11. Key features include:

* OpenMP parallelization for fitness evaluation
* REX crossover operator with V-shaped distribution
* Latin Hypercube Sampling (LHS) for initialization
* Equidistant selection crowding distance
* Batch evaluation support

Architecture
------------

The implementation follows SOLID principles with clear separation of concerns:

.. code-block:: cpp

   class NSGA2Optimizer {
   public:
       NSGA2Optimizer(const NSGA2Config& config);
       
       std::pair<std::vector<std::vector<double>>, 
                 std::vector<std::vector<double>>>
       optimize(const std::vector<ObjectiveFunction>& objectives,
               std::function<void(size_t, const Population&)> callback = nullptr,
               const BatchEvaluator& batch_evaluator = nullptr);
   };

Configuration
-------------

The optimizer is configured through the NSGA2Config structure:

.. code-block:: cpp

   struct NSGA2Config {
       size_t population_size = 100;
       size_t max_generations = 100;
       size_t n_objectives = 2;
       std::vector<double> lower_bounds;
       std::vector<double> upper_bounds;
       double crossover_prob = 0.9;
       double mutation_prob = 0.1;
       size_t n_parents = 3;
       size_t n_children = 10;
       unsigned int random_seed = 42;
       CrowdingDistanceType crowding_type = CrowdingDistanceType::Traditional;
       size_t progress_interval = 10;
   };

Non-dominated Sorting
---------------------

The non-dominated sorting algorithm has O(MN²) complexity:

.. code-block:: cpp

   void NSGA2Optimizer::non_dominated_sort(Population& population) {
       std::vector<std::vector<size_t>> fronts;
       std::vector<size_t> current_front;
       
       // For each individual p
       for (size_t p = 0; p < population.size(); ++p) {
           size_t n_p = 0;  // Number dominating p
           std::vector<size_t> S_p;  // Individuals dominated by p
           
           // Compare with every other individual q
           for (size_t q = 0; q < population.size(); ++q) {
               if (p == q) continue;
               
               auto dom = dominance(population[p], population[q]);
               if (dom == 1) {
                   S_p.push_back(q);  // p dominates q
               } else if (dom == -1) {
                   n_p++;  // q dominates p
               }
           }
           
           if (n_p == 0) {
               population[p]->rank = 1;
               current_front.push_back(p);
           }
           
           domination_count[p] = n_p;
           dominated_set[p] = S_p;
       }
   }

REX Crossover Operator
----------------------

The REX(φ,n+k) crossover uses a V-shaped distribution:

Mathematical Formulation
^^^^^^^^^^^^^^^^^^^^^^^^

The child is generated as:

.. math::

   \mathbf{x}_{\text{child}} = \mathbf{x}_g + \sum_{i=1}^{n_{\text{parents}}} \xi_i (\mathbf{x}_i - \mathbf{x}_g)

where:

* :math:`\mathbf{x}_g = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i` is the center of mass
* :math:`\xi_i` follows a V-shaped distribution with parameter :math:`a = \sqrt{2/n}`

V-shaped Distribution
^^^^^^^^^^^^^^^^^^^^^

The V-shaped distribution has PDF:

.. math::

   f(x) = \frac{|x|}{a^2}, \quad x \in [-a, a]

Implementation:

.. code-block:: cpp

   double REXCrossover::generateVShapedRandom(double a) {
       std::uniform_real_distribution<> dist(0.0, 1.0);
       double u = dist(gen_);
       
       if (u < 0.5) {
           // Left side: x = -a*sqrt(1-2u)
           return -a * std::sqrt(1.0 - 2.0 * u);
       } else {
           // Right side: x = a*sqrt(2u-1)
           return a * std::sqrt(2.0 * u - 1.0);
       }
   }

Crossover Implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   std::vector<IndividualPtr> REXCrossover::crossover(
       const std::vector<IndividualPtr>& parents,
       size_t n_children) {
       
       size_t n_dims = parents[0]->parameters.size();
       size_t n_parents = parents.size();
       std::vector<IndividualPtr> children;
       
       // Calculate center of mass
       std::vector<double> center(n_dims, 0.0);
       for (const auto& parent : parents) {
           for (size_t i = 0; i < n_dims; ++i) {
               center[i] += parent->parameters[i];
           }
       }
       for (double& c : center) {
           c /= n_parents;
       }
       
       // Generate children
       for (size_t c = 0; c < n_children; ++c) {
           auto child = std::make_shared<Individual>(n_dims);
           
           // Initialize with center
           for (size_t d = 0; d < n_dims; ++d) {
               child->parameters[d] = center[d];
           }
           
           // Add deviations using V-shaped distribution
           double a = std::sqrt(2.0 / n_parents);
           for (size_t p = 0; p < n_parents; ++p) {
               double xi = generateVShapedRandom(a);
               for (size_t d = 0; d < n_dims; ++d) {
                   child->parameters[d] += xi * 
                       (parents[p]->parameters[d] - center[d]);
               }
           }
           
           children.push_back(child);
       }
       
       return children;
   }

Latin Hypercube Sampling
------------------------

LHS ensures better coverage of the parameter space:

.. code-block:: cpp

   std::vector<std::vector<double>> LatinHypercubeSampler::sample(
       size_t n_samples, size_t n_dims,
       const std::vector<double>& lower_bounds,
       const std::vector<double>& upper_bounds) {
       
       std::vector<std::vector<double>> samples(n_samples, 
           std::vector<double>(n_dims));
       
       // For each dimension
       for (size_t d = 0; d < n_dims; ++d) {
           // Create permutation of intervals
           std::vector<size_t> perm(n_samples);
           std::iota(perm.begin(), perm.end(), 0);
           std::shuffle(perm.begin(), perm.end(), gen_);
           
           // Sample within each interval
           for (size_t i = 0; i < n_samples; ++i) {
               double lower = perm[i] / double(n_samples);
               double upper = (perm[i] + 1) / double(n_samples);
               
               std::uniform_real_distribution<> dist(lower, upper);
               double val = dist(gen_);
               
               // Scale to actual bounds
               samples[i][d] = lower_bounds[d] + 
                   val * (upper_bounds[d] - lower_bounds[d]);
           }
       }
       
       return samples;
   }

Crowding Distance
-----------------

Traditional Crowding Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   void TraditionalCrowdingDistance::calculate(
       std::vector<IndividualPtr>& front) {
       
       size_t front_size = front.size();
       size_t n_objectives = front[0]->objectives.size();
       
       // Initialize distances
       for (auto& ind : front) {
           ind->crowding_distance = 0.0;
       }
       
       // For each objective
       for (size_t m = 0; m < n_objectives; ++m) {
           // Sort by objective m
           std::sort(front.begin(), front.end(),
               [m](const IndividualPtr& a, const IndividualPtr& b) {
                   return a->objectives[m] < b->objectives[m];
               });
           
           // Boundary points get infinite distance
           front[0]->crowding_distance = std::numeric_limits<double>::infinity();
           front[front_size-1]->crowding_distance = 
               std::numeric_limits<double>::infinity();
           
           // Calculate distance for interior points
           double obj_range = front[front_size-1]->objectives[m] - 
                             front[0]->objectives[m];
           
           if (obj_range > 0) {
               for (size_t i = 1; i < front_size - 1; ++i) {
                   double distance = (front[i+1]->objectives[m] - 
                                    front[i-1]->objectives[m]) / obj_range;
                   front[i]->crowding_distance += distance;
               }
           }
       }
   }

Equidistant Selection
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   void EquidistantSelectionCrowdingDistance::calculate(
       std::vector<IndividualPtr>& front) {
       
       if (selection_size_ >= front.size()) {
           // All individuals selected
           for (auto& ind : front) {
               ind->crowding_distance = std::numeric_limits<double>::infinity();
           }
           return;
       }
       
       // Implementation of equidistant selection algorithm
       // ... (detailed implementation)
   }

Parallelization
---------------

OpenMP is used for parallel fitness evaluation:

.. code-block:: cpp

   void NSGA2Optimizer::evaluate_batch(
       std::vector<IndividualPtr>& individuals,
       const std::vector<ObjectiveFunction>& objectives,
       const BatchEvaluator& batch_evaluator) {
       
       if (batch_evaluator) {
           // Batch evaluation
           std::vector<std::vector<double>> param_batch;
           for (const auto& ind : individuals) {
               param_batch.push_back(ind->parameters);
           }
           
           auto objectives_batch = batch_evaluator(param_batch);
           
           for (size_t i = 0; i < individuals.size(); ++i) {
               individuals[i]->objectives = objectives_batch[i];
           }
       } else {
           // Individual evaluation (OpenMP parallelized)
           #pragma omp parallel for schedule(dynamic)
           for (size_t i = 0; i < individuals.size(); ++i) {
               individuals[i]->objectives.clear();
               for (const auto& obj_func : objectives) {
                   auto obj_values = obj_func(individuals[i]->parameters);
                   individuals[i]->objectives.insert(
                       individuals[i]->objectives.end(), 
                       obj_values.begin(), 
                       obj_values.end()
                   );
               }
           }
       }
   }

Python Bindings
---------------

The C++ implementation is exposed to Python using pybind11:

.. code-block:: cpp

   PYBIND11_MODULE(nsga2_optimizer, m) {
       m.doc() = "NSGA-II optimizer with Python bindings";
       
       // NSGA2Config
       py::class_<nsga2::NSGA2Config>(m, "NSGA2Config")
           .def(py::init<>())
           .def_readwrite("population_size", &nsga2::NSGA2Config::population_size)
           .def_readwrite("max_generations", &nsga2::NSGA2Config::max_generations)
           // ... other members
           ;
       
       // NSGA2Optimizer
       py::class_<nsga2::NSGA2Optimizer>(m, "NSGA2Optimizer")
           .def(py::init<const nsga2::NSGA2Config&>())
           .def("optimize", /* ... */)
           .def("get_pareto_front", /* ... */);
   }

Usage Example
-------------

Python usage:

.. code-block:: python

   import nsga2_optimizer
   
   # Configure optimizer
   config = nsga2_optimizer.NSGA2Config()
   config.population_size = 100
   config.max_generations = 1000
   config.n_objectives = 4
   config.lower_bounds = [-5.0] * 10
   config.upper_bounds = [5.0] * 10
   
   # Define objectives
   def objective_func(params):
       # Compute objectives
       return [obj1, obj2, obj3, obj4]
   
   # Create and run optimizer
   optimizer = nsga2_optimizer.NSGA2Optimizer(config)
   pareto_set, pareto_front = optimizer.optimize([objective_func])

Building from Source
--------------------

Requirements:

* C++17 compatible compiler
* OpenMP support
* Python 3.12+
* pybind11

Build commands:

.. code-block:: bash

   # Using setup.py
   python setup.py build_ext --inplace
   
   # Or using CMake
   mkdir build && cd build
   cmake ..
   make -j$(nproc)

Performance Considerations
--------------------------

1. **Population Size**: Larger populations explore more but increase computation
2. **Parallel Evaluation**: Enable OpenMP with sufficient cores
3. **Batch Evaluation**: Use for GPU-accelerated objective functions
4. **Memory Management**: Use shared_ptr for automatic memory management

References
----------

* Deb et al. (2002) "A fast and elitist multiobjective genetic algorithm: NSGA-II"
* Akimoto et al. (2018) "Adaptive Stochastic Natural Gradient Method"
* Ma et al. (2023) "A comprehensive survey on NSGA-II for multi-objective optimization"