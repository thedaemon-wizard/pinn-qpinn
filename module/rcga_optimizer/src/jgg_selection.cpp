#include "selection.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace rcga {

JGGSelection::JGGSelection(size_t num_parents)
    : num_parents_(num_parents) {
    if (num_parents < 2) {
        throw std::invalid_argument("JGG requires at least 2 parents");
    }
}

void JGGSelection::select(
    std::vector<IndividualPtr>& population,
    const std::vector<IndividualPtr>& offspring,
    std::mt19937& rng) {
    
    if (offspring.empty()) {
        return;
    }
    
    // Find the best offspring
    auto best_offspring_it = std::min_element(
        offspring.begin(), offspring.end(),
        [](const IndividualPtr& a, const IndividualPtr& b) {
            return a->getFitness() < b->getFitness();
        });
    
    // This call is already correct in the previous implementation
    // But let's make sure parent_indices is not empty
    std::vector<size_t> parent_indices = selectParentIndices(population.size(), rng);
    
    if (parent_indices.empty()) {
        throw std::runtime_error("No parent indices selected");
    }
    
    // Find worst parent among selected
    size_t worst_parent_idx = parent_indices[0];
    double worst_fitness = population[worst_parent_idx]->getFitness();
    
    for (size_t idx : parent_indices) {
        if (population[idx]->getFitness() > worst_fitness) {
            worst_fitness = population[idx]->getFitness();
            worst_parent_idx = idx;
        }
    }
    
    // Replace worst parent with best offspring
    population[worst_parent_idx] = (*best_offspring_it)->clone();
}

std::vector<size_t> JGGSelection::selectParentIndices(
    size_t population_size, std::mt19937& rng) {
    
    if (num_parents_ > population_size) {
        throw std::invalid_argument("Number of parents exceeds population size");
    }
    
    std::vector<size_t> indices(population_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    indices.resize(num_parents_);
    return indices;
}

} // namespace rcga