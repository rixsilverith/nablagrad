#ifndef NABLAGRAD_H
#define NABLAGRAD_H

#include <functional>
#include <vector>

#include "duals.hpp"

namespace nabla {
    using RealVec = std::vector<double>;
    using DualVec = std::vector<Dual>;

    // derivative a function 
    std::function<double(double)> derivative(std::function<Dual(Dual)> f);
    
    // gradient computation. F must be a function using nabla::Dual
    std::function<RealVec(const RealVec&)> grad(std::function<Dual(const DualVec&)> F);

    // directional derivative. F must be a nabla::Dual function
    std::function<double(const RealVec&, const RealVec&)> grad_dir(std::function<Dual(const DualVec&)> F);
}

#endif
