#include "tensor.hpp"
/* #include "tensor_ops.hpp" */
/* #include "core.hpp" */

#include "helpers.hpp"

#include <iostream>

namespace nabla {

// tensor base constructor
Tensor::Tensor(const std::string& name, const std::vector<size_t>& shape, bool requires_grad)
    : name_{name}, shape_{shape}, requires_grad_{requires_grad}
{
    stride_ = compute_stride_from_shape_(shape_);
    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    data_ = std::vector<double>(size_);
}

Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
    : Tensor(generate_default_name_(), shape, requires_grad) {}

Tensor Tensor::rand(const std::vector<size_t>& shape, bool requires_grad) {
    Tensor rand_tensor(shape, requires_grad);
    rand_tensor.data_ = generate_rand_vect<double>(rand_tensor.size());
    return rand_tensor;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
    Tensor zeros_tensor(shape, requires_grad);
    std::vector<double> tensor_data(zeros_tensor.size());
    fill(tensor_data.begin(), tensor_data.end(), 1);
    zeros_tensor.data_ = tensor_data;
    return zeros_tensor;
}

Tensor Tensor::ones(const std::vector<size_t>& shape, bool requires_grad) {
    Tensor ones_tensor(shape, requires_grad);
    std::vector<double> tensor_data(ones_tensor.size());
    fill(tensor_data.begin(), tensor_data.end(), 1);
    ones_tensor.data_ = tensor_data;
    return ones_tensor;
}

std::string Tensor::generate_default_name_() {
    return "tensor_" + std::to_string(tensor_next_id_++);
}

double& Tensor::at(const std::vector<size_t>& indices) {
    return this->data_[this->flatten_index_(indices)];
}

const double& Tensor::at(const std::vector<size_t>& indices) const {
    return this->data_[this->flatten_index_(indices)];
}

std::vector<size_t> Tensor::compute_stride_from_shape_(const std::vector<size_t>& shape) {
    std::vector<size_t> stride(shape.size());
    stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--)
        stride[i] = stride[i + 1] * shape[i + 1];
    return stride;
}

size_t Tensor::flatten_index_(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size())
        throw std::out_of_range("Incorrect tensor shape");

    size_t index = 0;
    size_t current_stride = 1;
    for (size_t i = indices.size() - 1; i >= 0; i--) {
        if (indices[i] >= shape_[i])
            throw std::out_of_range("Out of bounds");

        index += indices[i] * current_stride;
        current_stride *= shape_[i];
    }
    return index;
}

std::string Tensor::data_to_string_(
    const std::vector<size_t>& indices,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& stride,
    bool is_last
) const {
    std::string data_str = "[";
    if (shape.size() == 1) {
        for (size_t i = indices[0]; i <= indices[1]; i++) {
            data_str.append(std::to_string(data_[i]));
            if (i != indices[1]) data_str.append(", ");
        }
        data_str.append("]");
        if (!is_last) data_str.append(", ");
        return data_str;
    }

    size_t low_index = 0;
    size_t high_index = stride[0] - 1;
    std::vector<size_t> ushape(shape);
    std::vector<size_t> ustride(stride);
    ushape.erase(ushape.begin());
    ustride.erase(ustride.begin());

    for (size_t i = 0; i < shape[0]; i++) {
        if (i == shape[0] - 1) is_last = true;
        data_str.append(data_to_string_({low_index, high_index}, ushape, ustride, is_last));
        low_index += stride[0];
        high_index += stride[0];
    }
    data_str.append("]");
    return data_str;
}

std::ostream& operator<<(std::ostream& os, const Tensor& self) {
    /* os << "nabla::Tensor(" << tensor.name() << ", shape: " << tensor.shape() */
    /*    << ", data:\n" << tensor.to_string_() << ", requires_grad: " << tensor.requires_grad() << ")"; */
    /* return os; */
    os << "nabla::Tensor(" << self.name_ << ", data: " << self.to_string_()
       << ", requires_grad: " << self.requires_grad_ << ")";
    return os;
}

} // namespace nabla
