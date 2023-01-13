#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric> // std::accumulate

/* #include "gradient_tape.hpp" */

namespace nabla {
struct Tensor {
    /* Tensor( */
    /*     const std::string& name, std::vector<size_t> data, */
    /*     const std::vector<size_t>& shape, bool requires_grad=false */
    /* ); */
    /* Tensor(std::vector<double> data, const std::vector<size_t>& shape, bool grad=false) */
    /*     : data_{flatten_vec_(data)}, shape_{shape}, requires_grad_{grad} { */
    /*     if (data_.size() != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>())) */
    /*         throw std::invalid_argument("Data size does not match the shape of the tensor"); */

    /*     name_ = "tensor_" + std::to_string(tensor_next_id_++); */
    /*     compute_stride_from_shape(); */
    /* } */

    Tensor(const std::vector<size_t>& shape, bool requires_grad=false);
    Tensor(const std::string& name, const std::vector<size_t>& shape, bool requires_grad=false);

    static Tensor rand(const std::vector<size_t>& shape, bool grad=false);
    static Tensor zeros(const std::vector<size_t>& shape, bool grad=false);
    static Tensor ones(const std::vector<size_t>& shape, bool grad=false);

    double& at(const std::vector<size_t>& indices);
    const double& at(const std::vector<size_t>& indices) const;

    const std::string& name() const { return name_; }
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& stride() const { return stride_; }
    const std::vector<double>& data() const { return data_; }
    bool requires_grad() const { return requires_grad_; }
    size_t size() const { return size_; }

    Tensor flatten() {
        Tensor flat_tensor({1, data_.size()}, requires_grad_);
        flat_tensor.data_ = data_;
        return flat_tensor;
    }

    // string representation of the tensor data according to its shape
    // TODO: make this method private
    std::string to_string_() const {
        return data_to_string_({0, data_.size() - 1}, shape_, stride_, false);
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& self);

private:
    std::string generate_default_name_();
    std::vector<size_t> compute_stride_from_shape_(const std::vector<size_t>& shape);

    // TODO: refactor/clarify/document/provide some insight about how this works
    std::string data_to_string_(
        const std::vector<size_t>& indices,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& stride,
        bool is_last
    ) const;

    size_t flatten_index_(const std::vector<size_t>& indices) const;

    /* void compute_stride_from_shape(); */

    template<typename X>
    std::vector<double> flatten_vec_(const std::vector<X>& vec) {
        std::vector<double> f;
        flatten_helper(vec, f);
        return f;
    }

    std::string name_;
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    std::vector<double> data_;
    size_t size_;
    bool requires_grad_;

    static inline int tensor_next_id_ = 0;
};

} // namespace nabla

#endif
