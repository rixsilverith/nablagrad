#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.hpp"

#include <cmath>

namespace nabla {
    Tensor AddBackward(const Tensor& ltensor, const Tensor& rtensor);
    Tensor SubBackward(const Tensor& ltensor, const Tensor& rtensor);
    Tensor MultBackward(const Tensor& ltensor, const Tensor& rtensor);
    Tensor DivBackward(const Tensor& ltensor, const Tensor& rtensor);

    Tensor ExpBackward(const Tensor& tensor);
    Tensor LogBackward(const Tensor& tensor);
    Tensor PowerBackward(const Tensor& tensor, unsigned int power);

    Tensor SinBackward(const Tensor& tensor);
    Tensor CosBackward(const Tensor& tensor);

} // namespace nabla

#endif
