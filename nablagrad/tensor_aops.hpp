#include "tensor.hpp"

#include <cstdlib>
#include <functional>
#include <execution>
#include <cmath>
#include <algorithm>

namespace nabla {

    Tensor sin(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::sin(x); }); }
    Tensor cos(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::cos(x); }); }
    Tensor tan(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::tan(x); }); }
    Tensor log(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::log(x); }); }
    Tensor exp(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::exp(x); }); }

} // namespace nabla
