#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <numeric> // std::accumulate

#include "helpers.hpp"

/* #include "gradient_tape.hpp" */

namespace nabla {
    // constant to improve readabily when requiring gradient computation
    // at tensor creation; e.g.
    // nabla::Tensor::rand(..., nabla::require_grad);
    constexpr bool require_grad = true;

    struct Tensor {
        // ir means a tensor is an intermediate representation, which shouldnt be pushed into the
        // computation graph, as it will be pushed later. NOTE: this is just a quick dirty fix.
        // Will think about a more convinient way to do this later
        Tensor(const std::vector<size_t>& shape, bool requires_grad=false, bool ir=false);
        Tensor(const std::string& name, const std::vector<size_t>& shape, bool requires_grad=false, bool ir=false);
        Tensor() = default;

        static Tensor rand(const std::vector<size_t>& shape, bool grad=false);
        static Tensor zeros(const std::vector<size_t>& shape, bool grad=false);
        static Tensor ones(const std::vector<size_t>& shape, bool grad=false);

        double& at(const std::vector<size_t>& indices);
        const double& at(const std::vector<size_t>& indices) const;

        Tensor at(size_t index, size_t dim=0) const;

        Tensor t() const;

        const std::string& name() const { return name_; }
        const std::vector<size_t>& shape() const { return shape_; }
        size_t ndim() const { return shape_.size(); }
        const std::vector<size_t>& stride() const { return stride_; }
        const std::vector<double>& data() const { return data_; }
        bool requires_grad() const { return requires_grad_; }
        size_t size() const { return size_; }
        bool is_leaf() const { return is_leaf_; }

        Tensor flatten() {
            Tensor flat_tensor({data_.size()}, requires_grad_);
            flat_tensor.data_ = data_;
            return flat_tensor;
        }

        std::vector<double> raw_data() const { return data_; }
        void setdata(std::vector<double> v) { data_ = v; }

        // Apply the given transformation to the tensor elementwise.
        Tensor apply_transform(std::function<double(double)> transformation) const;
        void backward() const;

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

        size_t cg_node_idx_ = -1; // index of the tensor in the computation graph
        bool is_leaf_ = false;
        std::vector<double> grad_;
    private:
        std::string _generate_default_name_();
        std::vector<size_t> _compute_stride_from_shape_(const std::vector<size_t>& shape) const;

        // Given a set of indices referring to a position shaped tensor data, compute the
        // corresponding index in the plain internal 'data_' vector.
        size_t _flatten_index_(const std::vector<size_t>& indices) const;

        // Convert internal tensor data into a string representation according to its shape.
        // 'index0' represents the begin index of the current chunk of data being processed.
        // 'dim' is the current dimension (i.e. element of Tensor::shape_) which is being processed.
        // Both default to 0 and are passed down the recursion.
        std::string _data_to_string_(size_t index0=0, size_t dim=0) const;

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

#endif // TENSOR_H
