#pragma once

#include "individual.hpp"
#include <vector>
#include <random>

namespace rcga {

// Abstract crossover interface (Interface Segregation Principle)
class Crossover {
public:
    virtual ~Crossover() = default;
    virtual std::vector<IndividualPtr> cross(
        const std::vector<const Individual*>& parents,
        std::mt19937& rng) = 0;
};

// REX (Real-coded Crossover) implementation
class REXCrossover : public Crossover {
public:
    explicit REXCrossover(size_t num_children = 1, double xi = 1.0);
    
    std::vector<IndividualPtr> cross(
        const std::vector<const Individual*>& parents,
        std::mt19937& rng) override;
    
    void setXi(double xi) { xi_ = xi; }
    double getXi() const { return xi_; }
    
private:
    size_t num_children_;
    double xi_; // Expansion rate
};

} // namespace rcga