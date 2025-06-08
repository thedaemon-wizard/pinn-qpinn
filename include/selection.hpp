#pragma once

#include "individual.hpp"
#include <vector>
#include <random>

namespace rcga {

// Abstract selection interface
class Selection {
public:
    virtual ~Selection() = default;
    virtual void select(
        std::vector<IndividualPtr>& population,
        const std::vector<IndividualPtr>& offspring,
        std::mt19937& rng) = 0;
};

// JGG (Just Generation Gap) implementation
class JGGSelection : public Selection {
public:
    explicit JGGSelection(size_t num_parents = 2);
    
    void select(
        std::vector<IndividualPtr>& population,
        const std::vector<IndividualPtr>& offspring,
        std::mt19937& rng) override;
    
    std::vector<size_t> selectParentIndices(
        size_t population_size,
        std::mt19937& rng);
    
private:
    size_t num_parents_;
};

} // namespace rcga