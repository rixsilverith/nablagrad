#ifndef TENSOR_ALGEBRA_OPERATORS_H
#define TENSOR_ALGEBRA_OPERATORS_H

#include "tensor.hpp"
#include "autograd.hpp"

#include <cstdlib>
#include <functional>
#include <execution>
#include <cmath>
#include <algorithm>
#include <memory>

namespace nabla {

    // Backward pass for the `nabla::add()` tensor operator
    Tensor add_backward(const Tensor& self, const Tensor& other) {
        return Tensor::ones({2, self.size()});
    }

    Tensor add(const Tensor& self, const Tensor& other) {
        if (self.shape() != other.shape()) {
            std::cerr << "add: shapes of tensors differ" << std::endl;
        }

        ta_ops::TensorAdd add_op(self, other);
        Tensor out = add_op.forward();
        autograd::ComputationGraph::push_operator(add_op, out);
        return out;
    }

    Tensor mul(const Tensor& self, const Tensor& other) {
        Tensor add_tensor(self.shape(), self.requires_grad() || other.requires_grad());
        std::vector<double> res(self.raw_data());
        std::transform(res.begin(), res.end(), other.raw_data().begin(), res.begin(), std::multiplies<double>());
        add_tensor.setdata(res);
        return add_tensor;
    }

    Tensor sin(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::sin(x); }); }
    Tensor cos(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::cos(x); }); }
    Tensor tan(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::tan(x); }); }
    Tensor log(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::log(x); }); }
    Tensor exp(const Tensor& tensor) { return tensor.apply_transform([](double x) { return std::exp(x); }); }

} // namespace nabla

#endif // TENSOR_ALGEBRA_OPERATORS_H
