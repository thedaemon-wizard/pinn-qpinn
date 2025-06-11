#include "nsga2_optimizer.hpp"
#include <iostream>
#include <numeric>
#include <set>
#include <map>
#include <omp.h>
#include <chrono>
#include <algorithm>
#ifdef __cpp_lib_parallel_algorithm
#include <execution>
#endif

namespace nsga2 {

// Individual implementation
Individual::Individual(size_t n_params) 
    : parameters(n_params), rank(0), crowding_distance(0.0) {}

Individual::Individual(const std::vector<double>& params)
    : parameters(params), rank(0), crowding_distance(0.0) {}

bool Individual::dominates(const Individual& other) const {
    bool at_least_one_better = false;
    
    for (size_t i = 0; i < objectives.size(); ++i) {
        if (objectives[i] > other.objectives[i]) {
            return false;
        }
        if (objectives[i] < other.objectives[i]) {
            at_least_one_better = true;
        }
    }
    
    return at_least_one_better;
}

void Individual::reset() {
    rank = 0;
    crowding_distance = 0.0;
}

// Population implementation
Population::Population(size_t size, size_t n_params) {
    individuals_.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        individuals_.push_back(std::make_shared<Individual>(n_params));
    }
}

void Population::add(IndividualPtr ind) {
    individuals_.push_back(ind);
}

void Population::clear() {
    individuals_.clear();
}

size_t Population::size() const {
    return individuals_.size();
}

IndividualPtr& Population::operator[](size_t idx) {
    return individuals_[idx];
}

const IndividualPtr& Population::operator[](size_t idx) const {
    return individuals_[idx];
}

std::vector<IndividualPtr>::iterator Population::begin() {
    return individuals_.begin();
}

std::vector<IndividualPtr>::iterator Population::end() {
    return individuals_.end();
}

std::vector<IndividualPtr>::const_iterator Population::begin() const {
    return individuals_.begin();
}

std::vector<IndividualPtr>::const_iterator Population::end() const {
    return individuals_.end();
}

// Traditional crowding distance implementation
void TraditionalCrowdingDistance::calculate(std::vector<IndividualPtr>& front) {
    size_t n = front.size();
    if (n <= 2) {
        for (auto& ind : front) {
            ind->crowding_distance = std::numeric_limits<double>::infinity();
        }
        return;
    }
    
    // Initialize distances
    for (auto& ind : front) {
        ind->crowding_distance = 0.0;
    }
    
    size_t n_objectives = front[0]->objectives.size();
    
    for (size_t obj = 0; obj < n_objectives; ++obj) {
        // Sort by objective
        #ifdef __cpp_lib_parallel_algorithm
        std::sort(std::execution::par_unseq, front.begin(), front.end(),
            [obj](const IndividualPtr& a, const IndividualPtr& b) {
                return a->objectives[obj] < b->objectives[obj];
            });
        #else
        std::sort(front.begin(), front.end(),
            [obj](const IndividualPtr& a, const IndividualPtr& b) {
                return a->objectives[obj] < b->objectives[obj];
            });
        #endif
        
        // Boundary points
        front[0]->crowding_distance = std::numeric_limits<double>::infinity();
        front[n-1]->crowding_distance = std::numeric_limits<double>::infinity();
        
        double min_obj = front[0]->objectives[obj];
        double max_obj = front[n-1]->objectives[obj];
        double range = max_obj - min_obj;
        
        if (range == 0.0) continue;
        
        for (size_t i = 1; i < n - 1; ++i) {
            if (front[i]->crowding_distance != std::numeric_limits<double>::infinity()) {
                double distance = (front[i+1]->objectives[obj] - front[i-1]->objectives[obj]) / range;
                front[i]->crowding_distance += distance;
            }
        }
    }
}

// Equidistant selection crowding distance implementation
EquidistantSelectionCrowdingDistance::EquidistantSelectionCrowdingDistance(size_t selection_size)
    : selection_size_(selection_size) {}

double EquidistantSelectionCrowdingDistance::calculateDistance(const Individual& a, const Individual& b) const {
    double distance = 0.0;
    for (size_t i = 0; i < a.objectives.size(); ++i) {
        double diff = b.objectives[i] - a.objectives[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

void EquidistantSelectionCrowdingDistance::calculate(std::vector<IndividualPtr>& front) {
    if (front.empty() || front.size() < 2) return;
    
    size_t numSelect = (selection_size_ > 0) ? selection_size_ : front.size();
    
    // Sort by lexicographic order of objectives
    std::sort(front.begin(), front.end(), [](const IndividualPtr& a, const IndividualPtr& b) {
        for (size_t i = 0; i < a->objectives.size(); ++i) {
            if (std::abs(a->objectives[i] - b->objectives[i]) > std::numeric_limits<double>::epsilon()) {
                return a->objectives[i] < b->objectives[i];
            }
        }
        return false;
    });
    
    // Calculate path lengths
    std::vector<double> D(front.size(), 0.0);
    
    // Initialize crowding distances
    for (auto& ind : front) {
        ind->crowding_distance = 0.0;
    }
    
    // Calculate cumulative path length
    for (size_t i = 1; i < front.size(); ++i) {
        D[i] = D[i - 1] + calculateDistance(*front[i - 1], *front[i]);
    }
    
    double Dmax = D.back();
    if (Dmax < std::numeric_limits<double>::epsilon()) {
        std::cerr << "Warning: Total path length is near zero!" << std::endl;
        return;
    }
    
    // Set boundary points
    front[0]->crowding_distance = std::numeric_limits<double>::infinity();
    front.back()->crowding_distance = std::numeric_limits<double>::infinity();
    
    if (numSelect > 2) {
        double equidistantStep = Dmax / (numSelect - 1);
        
        for (size_t i = 1; i < numSelect - 1; ++i) {
            double targetDistance = equidistantStep * i;
            
            // Find closest point to target distance
            size_t bestIdx = 1;
            double minDiff = std::abs(D[1] - targetDistance);
            
            for (size_t j = 2; j < front.size() - 1; ++j) {
                double diff = std::abs(D[j] - targetDistance);
                if (diff < minDiff) {
                    minDiff = diff;
                    bestIdx = j;
                }
            }
            
            // Update crowding distance for selected point
            if (front[bestIdx]->crowding_distance == 0.0) {
                front[bestIdx]->crowding_distance = Dmax / numSelect;
            }
        }
    }
}

// CrowdingDistanceFactory implementation
std::unique_ptr<ICrowdingDistanceCalculator> CrowdingDistanceFactory::create(Type type, size_t selection_size) {
    switch (type) {
        case Type::Traditional:
            return std::make_unique<TraditionalCrowdingDistance>();
        case Type::EquidistantSelection:
            return std::make_unique<EquidistantSelectionCrowdingDistance>(selection_size);
        default:
            throw std::invalid_argument("Unknown crowding distance type");
    }
}

// FastNonDominatedSort implementation
std::vector<std::vector<IndividualPtr>> FastNonDominatedSort::sort(Population& population) {
    size_t n = population.size();
    std::vector<std::set<size_t>> S(n);
    std::vector<int> n_dominated(n, 0);
    std::vector<std::vector<size_t>> front_indices;
    std::vector<std::vector<IndividualPtr>> sorted_fronts;
    
    // Reset all individuals
    for (auto& ind : population) {
        ind->reset();
    }
    
    // Calculate domination relationships
    #pragma omp parallel for if(n > 100)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            
            if (population[i]->dominates(*population[j])) {
                #pragma omp critical
                S[i].insert(j);
            } else if (population[j]->dominates(*population[i])) {
                #pragma omp atomic
                n_dominated[i]++;
            }
        }
    }
    
    // Find first front
    std::vector<size_t> current_front;
    for (size_t i = 0; i < n; ++i) {
        if (n_dominated[i] == 0) {
            population[i]->rank = 0;
            current_front.push_back(i);
        }
    }
    
    front_indices.push_back(current_front);
    
    // Find remaining fronts
    int rank = 0;
    while (!current_front.empty()) {
        std::vector<size_t> next_front;
        
        for (size_t i : current_front) {
            for (size_t j : S[i]) {
                n_dominated[j]--;
                if (n_dominated[j] == 0) {
                    population[j]->rank = rank + 1;
                    next_front.push_back(j);
                }
            }
        }
        
        rank++;
        current_front = next_front;
        if (!current_front.empty()) {
            front_indices.push_back(current_front);
        }
    }
    
    // Convert indices to individual pointers
    sorted_fronts.resize(front_indices.size());
    for (size_t i = 0; i < front_indices.size(); ++i) {
        for (size_t idx : front_indices[i]) {
            sorted_fronts[i].push_back(population[idx]);
        }
    }
    
    return sorted_fronts;
}

// Latin Hypercube Sampling implementation
LatinHypercubeSampler::LatinHypercubeSampler(unsigned int seed) : gen_(seed) {}

std::vector<std::vector<double>> LatinHypercubeSampler::sample(
    size_t n_samples, size_t n_dims,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds) {
    
    std::vector<std::vector<double>> samples(n_samples, std::vector<double>(n_dims));
    
    for (size_t dim = 0; dim < n_dims; ++dim) {
        // Create intervals
        std::vector<size_t> intervals(n_samples);
        std::iota(intervals.begin(), intervals.end(), 0);
        std::shuffle(intervals.begin(), intervals.end(), gen_);
        
        // Sample within intervals
        std::uniform_real_distribution<> dist(0.0, 1.0);
        double range = upper_bounds[dim] - lower_bounds[dim];
        
        for (size_t i = 0; i < n_samples; ++i) {
            double interval_pos = (intervals[i] + dist(gen_)) / n_samples;
            samples[i][dim] = lower_bounds[dim] + interval_pos * range;
        }
    }
    
    return samples;
}

// REX Crossover implementation
REXCrossover::REXCrossover(double xi, unsigned int seed) 
    : xi_(xi), gen_(seed) {}

std::vector<IndividualPtr> REXCrossover::crossover(
    const std::vector<IndividualPtr>& parents,
    size_t n_children) {
    
    if (parents.size() < 2) {
        throw std::invalid_argument("REX crossover requires at least 2 parents");
    }
    
    size_t n_dims = parents[0]->parameters.size();
    std::vector<IndividualPtr> children;
    
    // Calculate center of mass
    std::vector<double> center(n_dims, 0.0);
    for (const auto& parent : parents) {
        for (size_t i = 0; i < n_dims; ++i) {
            center[i] += parent->parameters[i];
        }
    }
    for (double& c : center) {
        c /= parents.size();
    }
    
    // Generate children
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    for (size_t c = 0; c < n_children; ++c) {
        auto child = std::make_shared<Individual>(n_dims);
        
        // Generate weights
        std::vector<double> weights(parents.size());
        double sum_weights = 0.0;
        
        for (size_t i = 0; i < parents.size(); ++i) {
            weights[i] = dist(gen_);
            sum_weights += weights[i];
        }
        
        // Normalize weights
        for (double& w : weights) {
            w /= sum_weights;
        }
        
        // Create child
        for (size_t d = 0; d < n_dims; ++d) {
            child->parameters[d] = center[d];
            
            for (size_t p = 0; p < parents.size(); ++p) {
                child->parameters[d] += xi_ * weights[p] * 
                    (parents[p]->parameters[d] - center[d]);
            }
        }
        
        children.push_back(child);
    }
    
    return children;
}

// TournamentSelection implementation
bool TournamentSelection::crowding_operator(const IndividualPtr& a, const IndividualPtr& b) const {
    if (a->rank < b->rank) return true;
    if (a->rank > b->rank) return false;
    return a->crowding_distance > b->crowding_distance;
}

std::vector<IndividualPtr> TournamentSelection::select(const Population& population, 
                                                      size_t n_select,
                                                      std::mt19937& gen) {
    std::vector<IndividualPtr> selected;
    std::uniform_int_distribution<> dist(0, population.size() - 1);
    
    for (size_t i = 0; i < n_select; ++i) {
        size_t idx1 = dist(gen);
        size_t idx2 = dist(gen);
        
        while (idx1 == idx2) {
            idx2 = dist(gen);
        }
        
        if (crowding_operator(population[idx1], population[idx2])) {
            selected.push_back(population[idx1]);
        } else {
            selected.push_back(population[idx2]);
        }
    }
    
    return selected;
}

// NSGA2Optimizer implementation
NSGA2Optimizer::NSGA2Optimizer(const NSGA2Config& config)
    : config_(config), gen_(config.random_seed) {
    
    // Initialize components
    lhs_sampler_ = std::make_unique<LatinHypercubeSampler>(config.random_seed);
    rex_crossover_ = std::make_unique<REXCrossover>(config.rex_xi, config.random_seed + 1);
    crowding_calculator_ = CrowdingDistanceFactory::create(config.crowding_type, config.population_size);
    sorting_algorithm_ = std::make_unique<FastNonDominatedSort>();
    selection_strategy_ = std::make_unique<TournamentSelection>();
}

void NSGA2Optimizer::setCrowdingDistanceCalculator(std::unique_ptr<ICrowdingDistanceCalculator> calculator) {
    crowding_calculator_ = std::move(calculator);
}

void NSGA2Optimizer::setCrowdingDistanceCalculator(std::shared_ptr<ICrowdingDistanceCalculator> calculator) {
    // shared_ptrからunique_ptrを作成（コピーを作成）
    if (auto* traditional = dynamic_cast<TraditionalCrowdingDistance*>(calculator.get())) {
        crowding_calculator_ = std::make_unique<TraditionalCrowdingDistance>(*traditional);
    } else if (auto* equidistant = dynamic_cast<EquidistantSelectionCrowdingDistance*>(calculator.get())) {
        crowding_calculator_ = std::make_unique<EquidistantSelectionCrowdingDistance>(*equidistant);
    } else {
        throw std::invalid_argument("Unknown crowding distance calculator type");
    }
}

void NSGA2Optimizer::setSortingAlgorithm(std::unique_ptr<ISortingAlgorithm> algorithm) {
    sorting_algorithm_ = std::move(algorithm);
}

void NSGA2Optimizer::setSelectionStrategy(std::unique_ptr<ISelectionStrategy> strategy) {
    selection_strategy_ = std::move(strategy);
}

void NSGA2Optimizer::environmental_selection(Population& combined_pop, Population& new_pop) {
    auto sorted_fronts = sorting_algorithm_->sort(combined_pop);
    
    new_pop.clear();
    
    // Add fronts until population is filled
    for (auto& front : sorted_fronts) {
        if (new_pop.size() + front.size() <= config_.population_size) {
            for (auto& ind : front) {
                new_pop.add(ind);
            }
        } else {
            // Calculate crowding distance for last front
            crowding_calculator_->calculate(front);
            
            // Sort by crowding distance
            std::sort(front.begin(), front.end(),
                [](const IndividualPtr& a, const IndividualPtr& b) {
                    return a->crowding_distance > b->crowding_distance;
                });
            
            // Add individuals until population is full
            size_t remaining = config_.population_size - new_pop.size();
            for (size_t i = 0; i < remaining; ++i) {
                new_pop.add(front[i]);
            }
            break;
        }
    }
}

void NSGA2Optimizer::evaluate_batch(
    std::vector<IndividualPtr>& individuals,
    const std::vector<ObjectiveFunction>& objectives,
    const std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)>& batch_evaluator) {
    
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

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
NSGA2Optimizer::optimize(const std::vector<ObjectiveFunction>& objectives,
                        std::function<void(size_t, const Population&)> callback,
                        const std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)>& batch_evaluator) {
    
    // Initialize
    start_time_ = std::chrono::steady_clock::now();
    best_fitness_history_.clear();
    mean_fitness_history_.clear();
    best_fitness_ = std::numeric_limits<double>::infinity();
    current_generation_ = 0;
    
    size_t n_params = config_.lower_bounds.size();
    
    // Initialize population using LHS
    auto initial_params = lhs_sampler_->sample(
        config_.population_size, n_params,
        config_.lower_bounds, config_.upper_bounds
    );
    
    population_ = std::make_shared<Population>();
    for (const auto& params : initial_params) {
        auto ind = std::make_shared<Individual>(params);
        population_->add(ind);
    }
    
    // Evaluate initial population
    std::vector<IndividualPtr> initial_individuals;
    for (auto& ind : *population_) {
        initial_individuals.push_back(ind);
    }
    evaluate_batch(initial_individuals, objectives, batch_evaluator);
    
    // Main evolution loop
    for (size_t gen = 0; gen < config_.max_generations; ++gen) {
        current_generation_ = gen;
        
        // Calculate statistics
        double min_fitness = std::numeric_limits<double>::infinity();
        double sum_fitness = 0.0;
        
        for (const auto& ind : *population_) {
            double fitness = 0.0;
            std::vector<double> weights(objectives.size(), 1.0);
            
            for (size_t i = 0; i < ind->objectives.size() && i < weights.size(); ++i) {
                fitness += weights[i] * ind->objectives[i];
            }
            
            min_fitness = std::min(min_fitness, fitness);
            sum_fitness += fitness;
        }
        
        double mean_fitness = sum_fitness / population_->size();
        best_fitness_ = std::min(best_fitness_, min_fitness);
        best_fitness_history_.push_back(best_fitness_);
        mean_fitness_history_.push_back(mean_fitness);
        
        // Progress reporting
        if (config_.verbose && gen % config_.progress_interval == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time_
            ).count();
            
            std::cout << "\n--- NSGA-II Generation " << gen << " ---" << std::endl;
            std::cout << "Best fitness: " << min_fitness << ", Mean fitness: " << mean_fitness << std::endl;
            std::cout << "Elapsed time: " << elapsed << " seconds" << std::endl;
            std::cout << "Crowding distance method: " << crowding_calculator_->getName() << std::endl;
            
            if (gen > config_.progress_interval && best_fitness_history_.size() > config_.progress_interval) {
                double old_fitness = best_fitness_history_[gen - config_.progress_interval];
                double improvement = (old_fitness - min_fitness) / old_fitness * 100.0;
                std::cout << "Improvement (" << config_.progress_interval << " gen): " << improvement << "%" << std::endl;
            }
        }
        
        // Create offspring population
        Population offspring_pop;
        std::vector<IndividualPtr> all_children;
        
        while (offspring_pop.size() < config_.population_size) {
            // Select parents
            auto parents = selection_strategy_->select(*population_, config_.n_parents, gen_);
            
            // REX crossover
            auto children = rex_crossover_->crossover(parents, config_.n_children);
            
            // Apply bounds
            for (auto& child : children) {
                for (size_t i = 0; i < n_params; ++i) {
                    child->parameters[i] = std::max(config_.lower_bounds[i],
                        std::min(config_.upper_bounds[i], child->parameters[i]));
                }
                all_children.push_back(child);
                offspring_pop.add(child);
                
                if (offspring_pop.size() >= config_.population_size) {
                    break;
                }
            }
        }
        
        // Evaluate offspring
        evaluate_batch(all_children, objectives, batch_evaluator);
        
        // Combine parent and offspring populations
        Population combined_pop;
        for (auto& ind : *population_) {
            combined_pop.add(ind);
        }
        for (auto& ind : offspring_pop) {
            combined_pop.add(ind);
        }
        
        // Environmental selection
        Population new_population;
        environmental_selection(combined_pop, new_population);
        population_ = std::make_shared<Population>(std::move(new_population));
        
        // Callback
        if (callback) {
            callback(gen, *population_);
        }
    }
    
    // Extract final Pareto front
    auto pareto_front = get_pareto_front();
    
    std::vector<std::vector<double>> final_params;
    std::vector<std::vector<double>> final_objectives;
    
    for (const auto& ind : pareto_front) {
        final_params.push_back(ind->parameters);
        final_objectives.push_back(ind->objectives);
    }
    
    return {final_params, final_objectives};
}

std::vector<IndividualPtr> NSGA2Optimizer::get_pareto_front() const {
    std::vector<IndividualPtr> pareto_front;
    
    for (const auto& ind : *population_) {
        if (ind->rank == 0) {
            pareto_front.push_back(ind);
        }
    }
    
    return pareto_front;
}

} // namespace nsga2