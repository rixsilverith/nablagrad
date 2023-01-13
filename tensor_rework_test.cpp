#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <numeric> // std::acumulate
#include <random>
#include <chrono>

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

    static Tensor rand(const std::vector<size_t>& shape, bool grad=false) {
        unsigned int rseed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine rgen(rseed);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        size_t tensor_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        std::vector<double> tensor_data(tensor_size);

        for (size_t i = 0; i < tensor_size; i++)
            tensor_data[i] = U(rgen);

        return Tensor(tensor_data, shape, grad);
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
    bool requires_grad() const { return requires_grad_; }

    Tensor flatten() { return Tensor(data_, {1, data_.size()}, requires_grad_); }

    // string representation of the tensor data according to its shape
    // TODO: make this method private
    std::string to_string_() const {
        return data_to_string_({0, data_.size() - 1}, shape_, stride_, false);
    }

private:
    // TODO: refactor/clarify/document/provide some insight about how this works
    std::string data_to_string_(
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
    os << "nabla::Tensor(name: " << tensor.name() << ", shape: " << tensor.shape()
       << ", requires_grad: " << tensor.requires_grad() << ", data: " << tensor.to_string_() << ")";

    return os;
}

int main() {
    nabla::Tensor t = nabla::Tensor::rand({2, 2});
    std::cout << t << "\n";

    return 0;
}
