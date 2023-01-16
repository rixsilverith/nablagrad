#include "tensor.hpp"
/* #include "tensor_ops.hpp" */
/* #include "core.hpp" */

#include <iostream>
#include <iomanip>
#include <execution>
#include <algorithm> // std::reverse

namespace nabla {

    // tensor base constructor
    Tensor::Tensor(const std::string& name, const std::vector<size_t>& shape, bool requires_grad)
        : name_{name}, shape_{shape}, requires_grad_{requires_grad}
    {
        stride_ = _compute_stride_from_shape_(shape_);
        size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        data_ = std::vector<double>(size_);
    }

    Tensor::Tensor(const std::vector<size_t>& shape, bool requires_grad)
        : Tensor(_generate_default_name_(), shape, requires_grad) {}

    Tensor Tensor::rand(const std::vector<size_t>& shape, bool requires_grad) {
        Tensor rand_tensor(shape, requires_grad);
        rand_tensor.data_ = generate_rand_vect<double>(rand_tensor.size());
        return rand_tensor;
    }

    Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad) {
        Tensor zeros_tensor(shape, requires_grad);
        std::vector<double> tensor_data(zeros_tensor.size());
        fill(tensor_data.begin(), tensor_data.end(), 0);
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

    std::string Tensor::_generate_default_name_() {
        return "tensor_" + std::to_string(tensor_next_id_++);
    }

    double& Tensor::at(const std::vector<size_t>& indices) {
        return data_[_flatten_index_(indices)];
    }

    const double& Tensor::at(const std::vector<size_t>& indices) const {
        return data_[_flatten_index_(indices)];
    }

    Tensor Tensor::apply_transform(std::function<double(double)> transformation) const {
        std::vector<double> t_data(data_);
        std::transform(t_data.begin(), t_data.end(), t_data.begin(), [&transformation](double x) {
            return transformation(x); });

        Tensor tens(shape_, requires_grad_);
        tens.data_ = t_data;
        return tens;
    }

    Tensor Tensor::t() const {
        if (shape_.size() > 2)
            std::cerr << "t(): Cannot transpose a tensor with dimension > 2" << std::endl;

        if (shape_.size() == 1) {
            Tensor t_tensor(shape_, requires_grad_);
            t_tensor.data_ = data_;
            return t_tensor;
        }

        std::vector<size_t> t_shape(shape_);
        std::reverse(t_shape.begin(), t_shape.end());
        Tensor t_tensor(t_shape, requires_grad_);
        t_tensor.data_ = data_;
        return t_tensor;
    }

    std::vector<size_t> Tensor::_compute_stride_from_shape_(const std::vector<size_t>& shape) const {
        std::vector<size_t> stride(shape.size());
        stride[shape.size() - 1] = 1;
        for (int i = shape.size() - 2; i >= 0; i--)
            stride[i] = stride[i + 1] * shape[i + 1];
        return stride;
    }

    size_t Tensor::_flatten_index_(const std::vector<size_t>& indices) const {
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

    std::string Tensor::_data_to_string_(size_t index0, size_t dim) const {
        std::string data_str = "[";

        // Base case for recursion: innermost dimension
        if (dim == shape_.size() - 1) {
            // here 'shape_[dim]' is the number of elements in the current (innermost)
            // dimension. Since data is stored in 'row-major' format just iterate over the
            // base data vector interval [index0, index0 + shape_[dim] - 1]
            for (size_t i = index0; i <= index0 + shape_[dim] - 1; i++) {
                data_str.append(std::to_string(data_[i]));
                if (i != index0 + shape_[dim] - 1) data_str.append(", ");
            }
            data_str.append("]");
            return data_str;
        }

        // Iterate over each element in the current dimension
        for (size_t i = 0; i < shape_[dim]; i++) {
            data_str.append(_data_to_string_(index0, dim + 1));
            if (i != shape_[dim] - 1) {
                data_str.append("\n"); // each element in a separate line
                // for tensors with ndim > 2 each element in the current dimension is separated by
                // `(ndim - 2) - dim` blank lines (e.g. in a tensor with ndim = 3, elements in the
                // outermost dimension (dim = 0) are separated by one blank line)
                for (size_t j = dim; j < shape_.size() - 2; j++) data_str.append("\n");
                // add `dim + 1` spaces to the current element so that it is horizontally aligned
                // according to its dimension
                for (size_t j = 0; j <= dim; j++) data_str.append(" ");
            }
            index0 += stride_[dim]; // add offset between elements in the current dimension
        }
        data_str.append("]");
        return data_str;
    }

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << std::boolalpha; // true/false for booleans instead of 1/0
        os << "nabla::Tensor[shape: (";
        // For some reason printing 'tensor.shape_' directly leads to an infinite
        // recursion here. So we just manually print the 'tensor.shape_' vector.
        for (size_t i = 0; i < tensor.shape_.size(); i++) {
            os << tensor.shape_[i]; if (i != tensor.shape_.size() - 1) os << ", "; }
        os << "), requires_grad: " << tensor.requires_grad_ << "]\n";
        os << tensor._data_to_string_();
        return os;
    }

} // namespace nabla
