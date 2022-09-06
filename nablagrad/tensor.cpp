#include "tensor.hpp"
#include "tensor_ops.hpp"
#include "core.hpp"

#include <iostream>

namespace nabla {
    std::vector<double> Tensor::backward() const {
        std::cout << "[nabla::Tensor::backward] called on tensor " << *this << std::endl;

        node_index_t size = GradientTape::instance().get_tape().size();
        std::vector<double> gradients(size, 0);
        gradients.at(this->node_gradtape_index) = 1.0;

        for (node_index_t i = size - 1; i >= 0; --i) {
            const ComputationNode& node = GradientTape::instance().get_computation_node(i);

            // weight = localgrad for node * adjoint of the previous node
            double weight = node.get_local_grad().first * gradients.at(i);
            gradients.at(node.get_tensor_dependencies().first) += weight;

            weight = node.get_local_grad().second * gradients.at(i);
            gradients.at(node.get_tensor_dependencies().second) += weight;
        }

        return gradients;
    }

    Tensor operator*(const Tensor& ltensor, const Tensor& rtensor) {
        return MultBackward(ltensor, rtensor);
    }

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "nabla::Tensor[name: " << tensor.m_name << ", primal: " << tensor.m_primal
           << ", grad_tape_node_index: " << tensor.node_gradtape_index << "]";
        return os;
    }
} // namespace nabla
