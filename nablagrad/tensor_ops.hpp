#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "tensor.hpp"

namespace nabla {
    Tensor MultBackward(const Tensor& ltensor, const Tensor& rtensor); 
} // namespace nabla

#endif