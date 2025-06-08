#include "crossover.hpp"
#include <numeric>
#include <stdexcept>

namespace rcga {

REXCrossover::REXCrossover(size_t num_children, double xi)
    : num_children_(num_children), xi_(xi) {
    if (num_children == 0) {
        throw std::invalid_argument("Number of children must be greater than 0");
    }
}

std::vector<IndividualPtr> REXCrossover::cross(
    const std::vector<const Individual*>& parents,
    std::mt19937& rng) {
    
    if (parents.size() < 2) {
        throw std::invalid_argument("REX requires at least 2 parents");
    }
    
    // Check if all parents have the same dimension
    size_t dimension = parents[0]->getDimension();
    for (size_t i = 1; i < parents.size(); ++i) {
        if (parents[i]->getDimension() != dimension) {
            throw std::invalid_argument("All parents must have the same dimension");
        }
    }
    
    std::vector<IndividualPtr> offspring;
    offspring.reserve(num_children_);
    
    // Calculate center of mass
    std::vector<double> center(dimension, 0.0);
    for (const auto* parent : parents) {
        const auto& genes = parent->getGenes();
        for (size_t i = 0; i < dimension; ++i) {
            center[i] += genes[i];
        }
    }
    for (auto& c : center) {
        c /= static_cast<double>(parents.size());
    }
    
    // Generate offspring
    std::normal_distribution<> norm_dist(0.0, 1.0 / std::sqrt(static_cast<double>(parents.size())));
    
    for (size_t child_idx = 0; child_idx < num_children_; ++child_idx) {
        auto child = std::make_unique<Individual>(dimension);
        auto& child_genes = child->getGenes();
        
        // REX formula: child = center + xi * sum(alpha_i * (parent_i - center))
        for (size_t i = 0; i < dimension; ++i) {
            child_genes[i] = center[i];
            
            for (const auto* parent : parents) {
                double alpha = norm_dist(rng);
                child_genes[i] += xi_ * alpha * (parent->getGenes()[i] - center[i]);
            }
        }
        
        offspring.push_back(std::move(child));
    }
    
    return offspring;
}

} // namespace rcga