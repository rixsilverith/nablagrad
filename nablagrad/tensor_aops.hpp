#include "tensor.hpp"

#include <cstdlib>
#include <functional>
#include <execution>
#include <cmath>

namespace nabla {

/* Tensor elementwise_unary(const Tensor& tensor, std::function<double(double)> unary_op) { */
/*     std::vector<double> res(tensor.size()); */
/*     std::transform(std::execution::par_unseq, tensor.raw_data().begin(), tensor.raw_data().end(), res.begin(), unary_op); */
/*     Tensor res_tensor(tensor.shape(), tensor.requires_grad()); */
/*     res_tensor.setdata(res); */
/*     return res_tensor; */
/* } */

Tensor sin(const Tensor& tensor) {
    /* return elementwise_unary(tensor, std::sin); */
    std::vector<double> res(tensor.size());
    /* std::transform(std::execution::par_unseq, tensor.raw_data().begin(), tensor.raw_data().end(), res.begin(), [](double val) { return std::sin(val); }); */
    /* std::transform(tensor.raw_data().begin(), tensor.raw_data().end(), res.begin(), std::sin); */
    for (int i = 0; i < tensor.size(); i++) res[i] = std::sin(tensor.raw_data()[i]);
    Tensor res_tensor(tensor.shape(), tensor.requires_grad());
    res_tensor.setdata(res);
    return res_tensor;
}

Tensor log(const Tensor& tensor) {
    /* return elementwise_unary(tensor, std::sin); */
    std::vector<double> res(tensor.size());
    /* std::transform(std::execution::par_unseq, tensor.raw_data().begin(), tensor.raw_data().end(), res.begin(), [](double val) { return std::sin(val); }); */
    /* std::transform(tensor.raw_data().begin(), tensor.raw_data().end(), res.begin(), std::sin); */
    for (int i = 0; i < tensor.size(); i++) res[i] = std::log(tensor.raw_data()[i]);
    Tensor res_tensor(tensor.shape(), tensor.requires_grad());
    res_tensor.setdata(res);
    return res_tensor;
}

} // namespace nabla
