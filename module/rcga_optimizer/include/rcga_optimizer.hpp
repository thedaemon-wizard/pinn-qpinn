#pragma once

#include "individual.hpp"
#include "fitness_evaluator.hpp"
#include "crossover.hpp"
#include "selection.hpp"
#include <memory>
#include <vector>
#include <random>
#include <functional>

namespace rcga {

struct RCGAConfig {
    size_t population_size = 100;
    size_t max_generations = 1000;
    size_t num_parents = 2;
    size_t num_children = 5;
    double xi = 1.0;  // REX expansion rate
    double min_val = -3.14159;  // Parameter bounds
    double max_val = 3.14159;
    unsigned int random_seed = 42;
    bool verbose = false;
    bool use_lhs = true;  // Use Latin Hypercube Sampling for initialization
    size_t progress_interval = 50;  // Progress report interval
};

// Progress callback type
using ProgressCallback = std::function<void(
    size_t generation,
    double best_fitness,
    double mean_fitness,
    const std::vector<double>& best_solution
)>;

class RCGAOptimizer {
public:
    explicit RCGAOptimizer(const RCGAConfig& config = RCGAConfig());
    
    // Main optimization interface
    std::vector<double> optimize(
        size_t dimension,
        std::shared_ptr<FitnessEvaluator> evaluator);
    
    // Optimization with progress callback
    std::vector<double> optimize(
        size_t dimension,
        std::shared_ptr<FitnessEvaluator> evaluator,
        ProgressCallback progress_callback);
    
    // Getters
    const std::vector<double>& getBestSolution() const { return best_solution_; }
    double getBestFitness() const { return best_fitness_; }
    const std::vector<double>& getFitnessHistory() const { return fitness_history_; }
    const std::vector<double>& getMeanFitnessHistory() const { return mean_fitness_history_; }
    size_t getCurrentGeneration() const { return current_generation_; }
    
    // Configuration
    void setConfig(const RCGAConfig& config) { config_ = config; }
    const RCGAConfig& getConfig() const { return config_; }
    
private:
    void initializePopulation(size_t dimension);
    void initializePopulationLHS(size_t dimension);  // Latin Hypercube Sampling
    void evaluatePopulation();
    void updateBestSolution();
    double computeMeanFitness() const;
    std::vector<std::vector<double>> generateLatinHypercube(size_t n_samples, size_t dimension);
    
    RCGAConfig config_;
    std::mt19937 rng_;
    
    std::vector<IndividualPtr> population_;
    std::shared_ptr<FitnessEvaluator> evaluator_;
    std::unique_ptr<Crossover> crossover_;
    std::unique_ptr<Selection> selection_;
    ProgressCallback progress_callback_;
    
    std::vector<double> best_solution_;
    double best_fitness_;
    std::vector<double> fitness_history_;
    std::vector<double> mean_fitness_history_;
    size_t current_generation_;
};

} // namespace rcga