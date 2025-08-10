#pragma once

#include <vector>
#include <memory>
#include <random>

namespace rcga {

class Individual {
public:
    using GeneType = std::vector<double>;
    
    Individual(size_t dimension);
    Individual(const GeneType& genes);
    Individual(const Individual& other) = default;
    Individual(Individual&& other) = default;
    
    Individual& operator=(const Individual& other) = default;
    Individual& operator=(Individual&& other) = default;
    
    // Getters
    const GeneType& getGenes() const { return genes_; }
    GeneType& getGenes() { return genes_; }
    double getFitness() const { return fitness_; }
    size_t getDimension() const { return genes_.size(); }
    
    // Setters
    void setFitness(double fitness) { fitness_ = fitness; }
    void setGene(size_t index, double value);
    
    // Utility
    void randomize(double min_val, double max_val, std::mt19937& rng);
    std::unique_ptr<Individual> clone() const;
    
private:
    GeneType genes_;
    double fitness_;
};

using IndividualPtr = std::unique_ptr<Individual>;

} // namespace rcga