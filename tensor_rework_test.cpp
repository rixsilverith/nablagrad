#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <numeric> // std::acumulate

template<typename... T>
std::string strformat(const char *fmt, T... args) {
    const size_t n = snprintf(nullptr, 0, fmt, args...);
    std::vector<char> buf(n+1);
    snprintf(buf.data(), n+1, fmt, args...);
    return std::string(buf.data());
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); i++) {
        os << v[i];
        if (i != v.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

template <typename T, typename U>
void flatten_helper(const std::vector<T>& vec, std::vector<U>& flat) {
    flat.insert(flat.end(), vec.begin(), vec.end());
}

template <typename V, typename U>
void flatten_helper(const std::vector<std::vector<V>>& vec, std::vector<U>& flat) {
    for (const auto& x : vec) flatten_helper(x, flat);
}

namespace nabla {

struct Tensor {
    Tensor(std::vector<double> data, const std::vector<size_t>& shape, bool grad=false)
        : data_{flatten_vec_(data)}, shape_{shape}, requires_grad_{grad} {
        if (data_.size() != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()))
            throw std::invalid_argument("Data size does not match the shape of the tensor");

        name_ = "tensor_" + std::to_string(tensor_next_id_++);
        compute_stride_from_shape();
    }

    static Tensor ones(const std::vector<size_t>& shape, bool grad=false) {
        int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        std::vector<double> v(size);
        fill(v.begin(), v.end(), 1);
        return Tensor(v, shape, grad);
    }

    double& at(const std::vector<size_t> indices) {
        return data_[flatten_index(indices)];
    }

    const double& at(const std::vector<size_t> indices) const {
        return data_[flatten_index(indices)];
    }

    const std::string& name() const { return name_; }
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& stride() const { return stride_; }
    const std::vector<double>& data() const { return data_; }

    Tensor flatten() { return Tensor(data_, {1, data_.size()}, requires_grad_); }

private:
    size_t flatten_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size())
            throw std::out_of_range("Incorrect tensor shape");

        size_t index = 0;
        size_t stride = 1;
        for (size_t i = indices.size() - 1; i >= 0; i--) {
            if (indices[i] >= shape_[i])
                throw std::out_of_range(strformat("Index %d out of bounds for dimension %d (%d)",
                    indices[i], i, shape_[i]));
            index += indices[i] * stride;
            stride *= shape_[i];
        }
        return index;
    }

    void compute_stride_from_shape() {
        stride_.resize(shape_.size());
        stride_[shape_.size() - 1] = 1;
        for (int i = shape_.size() - 2; i >= 0; i--)
            stride_[i] = stride_[i + 1] * shape_[i + 1];
    }

    template<typename X>
    std::vector<double> flatten_vec_(const std::vector<X>& vec) {
        std::vector<double> f;
        flatten_helper(vec, f);
        return f;
    }

    std::vector<double> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    std::string name_;
    bool requires_grad_;

    static inline int tensor_next_id_ = 0;
};

}

std::ostream& operator<<(std::ostream& os, const nabla::Tensor& tensor) {
    os << "nabla::Tensor(" << tensor.data() << ", shape: " << tensor.shape()
        << ", name: " << tensor.name() << ")";
    return os;
}

int main() {
    nabla::Tensor x = nabla::Tensor::ones({2, 5, 3}, /*grad=*/true);
    std::cout << x << "\n";

    std::cout << x.at({1, 7, 2}) << std::endl;

    return 0;
}
