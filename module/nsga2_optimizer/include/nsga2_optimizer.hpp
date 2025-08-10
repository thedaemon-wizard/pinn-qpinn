#ifndef NSGA2_OPTIMIZER_HPP
#define NSGA2_OPTIMIZER_HPP

#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <string>
#include <stdexcept>

namespace nsga2 {

// Forward declarations
class Individual;
class Population;
class NSGA2Optimizer;

// Type aliases
using ObjectiveFunction = std::function<std::vector<double>(const std::vector<double>&)>;
using IndividualPtr = std::shared_ptr<Individual>;
using PopulationPtr = std::shared_ptr<Population>;

// Individual class representing a solution
class Individual {
public:
    std::vector<double> parameters;
    std::vector<double> objectives;
    int rank;
    double crowding_distance;
    
    Individual(size_t n_params);
    Individual(const std::vector<double>& params);
    
    bool dominates(const Individual& other) const;
    void reset();
};

// Population management class
class Population {
private:
    std::vector<IndividualPtr> individuals_;
    
public:
    Population() = default;
    Population(size_t size, size_t n_params);
    
    void add(IndividualPtr ind);
    void clear();
    size_t size() const;
    IndividualPtr& operator[](size_t idx);
    const IndividualPtr& operator[](size_t idx) const;
    
    std::vector<IndividualPtr>::iterator begin();
    std::vector<IndividualPtr>::iterator end();
    std::vector<IndividualPtr>::const_iterator begin() const;
    std::vector<IndividualPtr>::const_iterator end() const;
};

// Abstract interface for crowding distance calculation (Strategy Pattern)
class ICrowdingDistanceCalculator {
public:
    virtual ~ICrowdingDistanceCalculator() = default;
    virtual void calculate(std::vector<IndividualPtr>& front) = 0;
    virtual std::string getName() const = 0;
};

// Traditional crowding distance calculator
class TraditionalCrowdingDistance : public ICrowdingDistanceCalculator {
public:
    TraditionalCrowdingDistance() = default;
    TraditionalCrowdingDistance(const TraditionalCrowdingDistance&) = default;
    void calculate(std::vector<IndividualPtr>& front) override;
    std::string getName() const override { return "Traditional"; }
};

// Equidistant selection crowding distance calculator
class EquidistantSelectionCrowdingDistance : public ICrowdingDistanceCalculator {
private:
    size_t selection_size_;
    
public:
    explicit EquidistantSelectionCrowdingDistance(size_t selection_size = 0);
    EquidistantSelectionCrowdingDistance(const EquidistantSelectionCrowdingDistance& other) = default;
    void setSelectionSize(size_t size) { selection_size_ = size; }
    void calculate(std::vector<IndividualPtr>& front) override;
    std::string getName() const override { return "EquidistantSelection"; }
    
private:
    double calculateDistance(const Individual& a, const Individual& b) const;
};

// Factory for creating crowding distance calculators
class CrowdingDistanceFactory {
public:
    enum class Type {
        Traditional,
        EquidistantSelection
    };
    
    static std::unique_ptr<ICrowdingDistanceCalculator> create(Type type, size_t selection_size = 0);
};

// Abstract interface for sorting algorithms
class ISortingAlgorithm {
public:
    virtual ~ISortingAlgorithm() = default;
    virtual std::vector<std::vector<IndividualPtr>> sort(Population& population) = 0;
};

// Fast non-dominated sorting implementation
class FastNonDominatedSort : public ISortingAlgorithm {
public:
    std::vector<std::vector<IndividualPtr>> sort(Population& population) override;
};

// Latin Hypercube Sampling for initialization
class LatinHypercubeSampler {
private:
    std::mt19937 gen_;
    
public:
    LatinHypercubeSampler(unsigned int seed = 42);
    std::vector<std::vector<double>> sample(size_t n_samples, size_t n_dims,
                                           const std::vector<double>& lower_bounds,
                                           const std::vector<double>& upper_bounds);
};

// REX (Real-coded Ensemble Crossover) operator
class REXCrossover {
private:
    double xi_;  // Extension rate parameter
    std::mt19937 gen_;
    
public:
    REXCrossover(double xi = 1.2, unsigned int seed = 42);
    std::vector<IndividualPtr> crossover(const std::vector<IndividualPtr>& parents,
                                        size_t n_children);
};

// NSGA-II optimizer configuration
struct NSGA2Config {
    size_t population_size = 100;
    size_t max_generations = 1000;
    size_t n_objectives = 2;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    double rex_xi = 1.2;
    size_t n_parents = 3;
    size_t n_children = 10;
    unsigned int random_seed = 42;
    bool verbose = false;
    size_t progress_interval = 50;
    CrowdingDistanceFactory::Type crowding_type = CrowdingDistanceFactory::Type::Traditional;
};

// Selection strategy interface
class ISelectionStrategy {
public:
    virtual ~ISelectionStrategy() = default;
    virtual std::vector<IndividualPtr> select(const Population& population, 
                                             size_t n_select,
                                             std::mt19937& gen) = 0;
};

// Tournament selection implementation
class TournamentSelection : public ISelectionStrategy {
public:
    std::vector<IndividualPtr> select(const Population& population, 
                                    size_t n_select,
                                    std::mt19937& gen) override;
    
private:
    bool crowding_operator(const IndividualPtr& a, const IndividualPtr& b) const;
};

// Main NSGA-II optimizer class
class NSGA2Optimizer {
private:
    NSGA2Config config_;
    PopulationPtr population_;
    std::mt19937 gen_;
    
    // Strategy pattern components
    std::unique_ptr<LatinHypercubeSampler> lhs_sampler_;
    std::unique_ptr<REXCrossover> rex_crossover_;
    std::unique_ptr<ICrowdingDistanceCalculator> crowding_calculator_;
    std::unique_ptr<ISortingAlgorithm> sorting_algorithm_;
    std::unique_ptr<ISelectionStrategy> selection_strategy_;
    
    // Core NSGA-II operations
    void environmental_selection(Population& combined_pop, Population& new_pop);
    
    // Statistics tracking
    std::chrono::steady_clock::time_point start_time_;
    std::vector<double> best_fitness_history_;
    std::vector<double> mean_fitness_history_;
    double best_fitness_;
    size_t current_generation_;
    
public:
    NSGA2Optimizer(const NSGA2Config& config);
    
    // Setter methods for strategy pattern (Open/Closed Principle)
    void setCrowdingDistanceCalculator(std::unique_ptr<ICrowdingDistanceCalculator> calculator);
    void setCrowdingDistanceCalculator(std::shared_ptr<ICrowdingDistanceCalculator> calculator);
    void setSortingAlgorithm(std::unique_ptr<ISortingAlgorithm> algorithm);
    void setSelectionStrategy(std::unique_ptr<ISelectionStrategy> strategy);
    
    // Main optimization method
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    optimize(const std::vector<ObjectiveFunction>& objectives,
            std::function<void(size_t, const Population&)> callback = nullptr,
            const std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)>& batch_evaluator = nullptr);
    
    // Get Pareto front
    std::vector<IndividualPtr> get_pareto_front() const;
    
    // History accessors
    const std::vector<double>& get_fitness_history() const { return best_fitness_history_; }
    const std::vector<double>& get_mean_fitness_history() const { return mean_fitness_history_; }
    double get_best_fitness() const { return best_fitness_; }
    size_t get_current_generation() const { return current_generation_; }
    
    // Batch evaluation function
    void evaluate_batch(
        std::vector<IndividualPtr>& individuals,
        const std::vector<ObjectiveFunction>& objectives,
        const std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&)>& batch_evaluator
    );
};

} // namespace nsga2

#endif // NSGA2_OPTIMIZER_HPP