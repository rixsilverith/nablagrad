#include "gradient_tape.hpp"

namespace nabla {
    ComputationNode::ComputationNode(const std::string& node_name,
                                     std::pair<double, double> local_grad,
                                     std::pair<node_index_t, node_index_t> tensor_indices) 
        : m_node_name{node_name}, m_local_grad{local_grad}, m_tensor_indices{tensor_indices} {}

    node_index_t GradientTape::push_node(
        const std::string& node_name,
        const std::pair<double, double>& local_grad,
        const std::pair<node_index_t, node_index_t>& tensor_indices
    ) {
        size_t gradient_tape_size = this->tape.size();
        this->tape.emplace_back(ComputationNode(node_name, local_grad, tensor_indices));
        return gradient_tape_size;
    }

    node_index_t GradientTape::push_node(const std::string& node_name, double weight, node_index_t tensor_index) {
        size_t gradient_tape_size = this->tape.size();
        return this->push_node(node_name, { weight, 0. }, { tensor_index, gradient_tape_size });
    }

    node_index_t GradientTape::push_leaf_node(const std::string& node_name) {
        size_t gradient_tape_size = this->tape.size();
        std::string name = "leaf_" + std::to_string(ComputationNode::node_id_generator++);
        return this->push_node(name, { 0., 0. }, { gradient_tape_size, gradient_tape_size });
    }
} // namespace nabla