#include "rcga_optimizer.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <iomanip>

namespace rcga {

Individual::Individual(size_t dimension)
    : genes_(dimension, 0.0), fitness_(std::numeric_limits<double>::max()) {}

Individual::Individual(const GeneType& genes)
    : genes_(genes), fitness_(std::numeric_limits<double>::max()) {}

void Individual::setGene(size_t index, double value) {
    if (index < genes_.size()) {
        genes_[index] = value;
    }
}

void Individual::randomize(double min_val, double max_val, std::mt19937& rng) {
    std::uniform_real_distribution<> dist(min_val, max_val);
    for (auto& gene : genes_) {
        gene = dist(rng);
    }
}

std::unique_ptr<Individual> Individual::clone() const {
    return std::make_unique<Individual>(*this);
}

// FitnessEvaluator implementations
std::vector<double> FitnessEvaluator::evaluateBatch(const std::vector<IndividualPtr>& individuals) {
    std::vector<double> fitnesses;
    fitnesses.reserve(individuals.size());
    for (const auto& ind : individuals) {
        fitnesses.push_back(evaluate(*ind));
    }
    return fitnesses;
}

PythonFitnessEvaluator::PythonFitnessEvaluator(EvalFunction eval_func)
    : eval_func_(eval_func) {}

PythonFitnessEvaluator::PythonFitnessEvaluator(
    EvalFunction eval_func, BatchEvalFunction batch_eval_func)
    : eval_func_(eval_func), batch_eval_func_(batch_eval_func) {}

double PythonFitnessEvaluator::evaluate(const Individual& individual) {
    return eval_func_(individual.getGenes());
}

std::vector<double> PythonFitnessEvaluator::evaluateBatch(const std::vector<IndividualPtr>& individuals) {
    if (batch_eval_func_) {
        std::vector<std::vector<double>> genes_batch;
        genes_batch.reserve(individuals.size());
        for (const auto& ind : individuals) {
            genes_batch.push_back(ind->getGenes());
        }
        return batch_eval_func_(genes_batch);
    }
    return FitnessEvaluator::evaluateBatch(individuals);
}

// RCGAOptimizer implementation
RCGAOptimizer::RCGAOptimizer(const RCGAConfig& config)
    : config_(config), rng_(config.random_seed), 
      best_fitness_(std::numeric_limits<double>::max()),
      current_generation_(0) {
    crossover_ = std::make_unique<REXCrossover>(config.num_children, config.xi);
    selection_ = std::make_unique<JGGSelection>(config.num_parents);
}

std::vector<double> RCGAOptimizer::optimize(
    size_t dimension, std::shared_ptr<FitnessEvaluator> evaluator) {
    return optimize(dimension, evaluator, nullptr);
}

std::vector<double> RCGAOptimizer::optimize(
    size_t dimension, 
    std::shared_ptr<FitnessEvaluator> evaluator,
    ProgressCallback progress_callback) {
    
    evaluator_ = evaluator;
    progress_callback_ = progress_callback;
    fitness_history_.clear();
    mean_fitness_history_.clear();
    current_generation_ = 0;
    
    // Initialize population with LHS or random
    if (config_.use_lhs) {
        initializePopulationLHS(dimension);
        if (config_.verbose) {
            std::cout << "初期集団をLatin Hypercube Samplingで生成しました" << std::endl;
        }
    } else {
        initializePopulation(dimension);
        if (config_.verbose) {
            std::cout << "初期集団をランダムに生成しました" << std::endl;
        }
    }
    
    evaluatePopulation();
    updateBestSolution();
    
    // Initial progress report
    if (config_.verbose || progress_callback_) {
        double mean_fitness = computeMeanFitness();
        mean_fitness_history_.push_back(mean_fitness);
        
        if (config_.verbose) {
            std::cout << "世代 0: 最良適応度 = " << std::fixed << std::setprecision(6) 
                     << best_fitness_ << ", 平均適応度 = " << mean_fitness << std::endl;
        }
        
        if (progress_callback_) {
            progress_callback_(0, best_fitness_, mean_fitness, best_solution_);
        }
    }
    
    // Main evolutionary loop
    for (size_t gen = 1; gen <= config_.max_generations; ++gen) {
        current_generation_ = gen;
        
        // Select parents using JGG
        auto jgg_selection = dynamic_cast<JGGSelection*>(selection_.get());
        auto parent_indices = jgg_selection->selectParentIndices(population_.size(), rng_);
        
        // Gather parent pointers
        std::vector<const Individual*> parents;
        for (size_t idx : parent_indices) {
            parents.push_back(population_[idx].get());
        }
        
        // Generate offspring using REX
        auto offspring = crossover_->cross(parents, rng_);
        
        // Evaluate offspring
        if (evaluator_->supportsBatchEvaluation()) {
            auto fitnesses = evaluator_->evaluateBatch(offspring);
            for (size_t i = 0; i < offspring.size(); ++i) {
                offspring[i]->setFitness(fitnesses[i]);
            }
        } else {
            for (auto& child : offspring) {
                child->setFitness(evaluator_->evaluate(*child));
            }
        }
        
        // Selection (JGG replaces worst parent with best offspring)
        selection_->select(population_, offspring, rng_);
        
        // Update best solution
        updateBestSolution();
        fitness_history_.push_back(best_fitness_);
        
        // Compute and store mean fitness
        double mean_fitness = computeMeanFitness();
        mean_fitness_history_.push_back(mean_fitness);
        
        // Progress report
        if ((config_.verbose && gen % config_.progress_interval == 0) || 
            (progress_callback_ && gen % config_.progress_interval == 0)) {
            
            if (config_.verbose) {
                std::cout << "世代 " << gen << ": 最良適応度 = " 
                         << std::fixed << std::setprecision(6) << best_fitness_ 
                         << ", 平均適応度 = " << mean_fitness;
                
                // Show improvement
                if (fitness_history_.size() > config_.progress_interval) {
                    double old_fitness = fitness_history_[fitness_history_.size() - config_.progress_interval - 1];
                    double improvement = (old_fitness - best_fitness_) / old_fitness * 100.0;
                    std::cout << " (改善率: " << std::fixed << std::setprecision(2) 
                             << improvement << "%)";
                }
                std::cout << std::endl;
            }
            
            if (progress_callback_) {
                progress_callback_(gen, best_fitness_, mean_fitness, best_solution_);
            }
        }
    }
    
    // Final report
    if (config_.verbose) {
        std::cout << "\nRCGA最適化完了" << std::endl;
        std::cout << "最終世代: " << current_generation_ << std::endl;
        std::cout << "最良適応度: " << std::fixed << std::setprecision(6) 
                 << best_fitness_ << std::endl;
        
        if (!fitness_history_.empty()) {
            double initial_fitness = fitness_history_[0];
            double improvement = (initial_fitness - best_fitness_) / initial_fitness * 100.0;
            std::cout << "総改善率: " << std::fixed << std::setprecision(2) 
                     << improvement << "%" << std::endl;
        }
    }
    
    return best_solution_;
}

void RCGAOptimizer::initializePopulation(size_t dimension) {
    population_.clear();
    population_.reserve(config_.population_size);
    
    for (size_t i = 0; i < config_.population_size; ++i) {
        auto ind = std::make_unique<Individual>(dimension);
        ind->randomize(config_.min_val, config_.max_val, rng_);
        population_.push_back(std::move(ind));
    }
}

void RCGAOptimizer::initializePopulationLHS(size_t dimension) {
    population_.clear();
    population_.reserve(config_.population_size);
    
    // Generate Latin Hypercube samples
    auto lhs_samples = generateLatinHypercube(config_.population_size, dimension);
    
    // Create individuals from LHS samples
    for (const auto& sample : lhs_samples) {
        auto ind = std::make_unique<Individual>(sample);
        population_.push_back(std::move(ind));
    }
}

std::vector<std::vector<double>> RCGAOptimizer::generateLatinHypercube(
    size_t n_samples, size_t dimension) {
    
    std::vector<std::vector<double>> samples(n_samples, std::vector<double>(dimension));
    
    // For each dimension
    for (size_t d = 0; d < dimension; ++d) {
        // Create permutation
        std::vector<size_t> permutation(n_samples);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), rng_);
        
        // Generate samples for this dimension
        std::uniform_real_distribution<> dist(0.0, 1.0);
        
        for (size_t i = 0; i < n_samples; ++i) {
            // Random value within the cell
            double u = dist(rng_);
            
            // Latin hypercube sample in [0, 1]
            double sample_01 = (permutation[i] + u) / n_samples;
            
            // Scale to [min_val, max_val]
            samples[i][d] = config_.min_val + sample_01 * (config_.max_val - config_.min_val);
        }
    }
    
    return samples;
}

void RCGAOptimizer::evaluatePopulation() {
    if (evaluator_->supportsBatchEvaluation()) {
        auto fitnesses = evaluator_->evaluateBatch(population_);
        for (size_t i = 0; i < population_.size(); ++i) {
            population_[i]->setFitness(fitnesses[i]);
        }
    } else {
        for (auto& ind : population_) {
            ind->setFitness(evaluator_->evaluate(*ind));
        }
    }
}

void RCGAOptimizer::updateBestSolution() {
    for (const auto& ind : population_) {
        if (ind->getFitness() < best_fitness_) {
            best_fitness_ = ind->getFitness();
            best_solution_ = ind->getGenes();
        }
    }
}

double RCGAOptimizer::computeMeanFitness() const {
    double sum = 0.0;
    for (const auto& ind : population_) {
        sum += ind->getFitness();
    }
    return sum / population_.size();
}

} // namespace rcga