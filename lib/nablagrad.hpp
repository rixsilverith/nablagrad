#ifndef NABLAGRAD_H
#define NABLAGRAD_H

#include "duals.hpp"
#include <functional>
#include <vector>

namespace nabla {

std::function<double(double)> derivative(std::function<Dual(Dual)> f);

using RealVec = std::vector<double>;
using DualVec = std::vector<Dual>;

std::function<std::vector<double>(const std::vector<double>&)> grad(std::function<Dual(const std::vector<Dual>&)> F);
std::function<double(const std::vector<double>&, const std::vector<double>&)> grad_dir(
    std::function<Dual(const std::vector<Dual>&)> F
);

}

#endif
