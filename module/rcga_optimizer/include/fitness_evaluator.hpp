#pragma once

#include "individual.hpp"
#include <functional>
#include <vector>

namespace rcga {

// Abstract base class for fitness evaluation (Dependency Inversion Principle)
class FitnessEvaluator {
public:
    virtual ~FitnessEvaluator() = default;
    virtual double evaluate(const Individual& individual) = 0;
    virtual std::vector<double> evaluateBatch(const std::vector<IndividualPtr>& individuals);
    virtual bool supportsBatchEvaluation() const { return false; }
};

// Concrete implementation for Python callback
class PythonFitnessEvaluator : public FitnessEvaluator {
public:
    using EvalFunction = std::function<double(const std::vector<double>&)>;
    using BatchEvalFunction = std::function<std::vector<double>(const std::vector<std::vector<double>>&)>;
    
    explicit PythonFitnessEvaluator(EvalFunction eval_func);
    PythonFitnessEvaluator(EvalFunction eval_func, BatchEvalFunction batch_eval_func);
    
    double evaluate(const Individual& individual) override;
    std::vector<double> evaluateBatch(const std::vector<IndividualPtr>& individuals) override;
    bool supportsBatchEvaluation() const override { return batch_eval_func_ ? true : false; }
    
private:
    EvalFunction eval_func_;
    BatchEvalFunction batch_eval_func_;
};

} // namespace rcga