#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric> // std::accumulate

#include "helpers.hpp"

/* #include "gradient_tape.hpp" */

namespace nabla {

// constant to improve readabily when requiring gradient computation
// at tensor creation; e.g.
// nabla::Tensor::rand(..., nabla::require_grad);
constexpr bool require_grad = true;

struct Tensor {
    Tensor(const std::vector<size_t>& shape, bool requires_grad=false);
    Tensor(const std::string& name, const std::vector<size_t>& shape, bool requires_grad=false);

    static Tensor rand(const std::vector<size_t>& shape, bool grad=false);
    static Tensor zeros(const std::vector<size_t>& shape, bool grad=false);
    static Tensor ones(const std::vector<size_t>& shape, bool grad=false);

    double& at(const std::vector<size_t>& indices);
    const double& at(const std::vector<size_t>& indices) const;

    Tensor t() const;

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

    std::string to_string_() const {
        // here we need to initialize the recursive call with the outermost
        // first chunk of data, that is, [0, stride[0]-1]
        // the internal call will manage to traverse the whole data array
        // from the shape
        return data_to_string_(0, {0, stride_[0] - 1}, shape_, stride_, shape_.size() == 1);
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    std::vector<double> raw_data() const { return data_; }
    void setdata(std::vector<double> v) { data_ = v; }

private:
    std::string generate_default_name_();
    std::vector<size_t> compute_stride_from_shape_(const std::vector<size_t>& shape) const;
    size_t flatten_index_(const std::vector<size_t>& indices) const;

    std::string data_to_string_(
            size_t low_index,
        const std::vector<size_t>& indices,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& stride,
        bool is_last, size_t dim=0
    ) const;

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
