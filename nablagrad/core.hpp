#ifndef CORE_H
#define CORE_H

#include <iostream>
#include <functional>
#include <vector>

#include "dual.hpp"
#include "tensor.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i]; if (i != v.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

namespace nabla {
    using RealVec = std::vector<double>;
    using Gradient = std::vector<double>;
    using DualVec = std::vector<Dual>;
    using TensorVec = std::vector<Tensor>;

    // gradient computation using reverse-mode
    std::function<Gradient(const RealVec&)> grad(std::function<Tensor(const TensorVec&)> f);

    // derivative a single-valued function f:R->R
    std::function<double(double)> grad_forward(std::function<Dual(Dual)> f);
    
    // gradient computation. F must be a function using nabla::Dual (f:R^n->R)
    std::function<RealVec(const RealVec&)> grad_forward(std::function<Dual(const DualVec&)> F);
    
    // gradient of f evaluated at x
    std::vector<double> grad_forward(std::function<Dual(const DualVec&)> f, const RealVec& x);

    // directional derivative. F must be a nabla::Dual function
    std::function<double(const RealVec&, const RealVec&)> grad_dir(std::function<Dual(const DualVec&)> F);
}

#endif
