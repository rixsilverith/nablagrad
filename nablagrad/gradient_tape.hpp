#ifndef GRADIENT_TAPE_H
#define GRADIENT_TAPE_H

#include <iostream>
#include <vector>

namespace nabla {
    using node_index_t = int32_t;

    // Node in the computational graph used for tracing a tensor operation. Each
    // node stores the indices of the input tensors in the gradient tape, its
    // associated tensor (reference or value, depending on whether is a leaf node
    // or not) and the local gradient of the computation node, that is, the
    // corresponding weights for the dependency tensor operands.
    struct ComputationNode {
        ComputationNode(const std::string& node_name,
                        std::pair<double, double> local_grad,
                        std::pair<node_index_t, node_index_t> tensor_indices);

        const std::string& get_name() const { return this->m_node_name; }
        const std::pair<double, double>& get_local_grad() const { return this->m_local_grad; }
        const std::pair<node_index_t, node_index_t>& get_tensor_dependencies() const {
            return this->m_tensor_indices;
        }

        friend std::ostream& operator<<(std::ostream& os, const ComputationNode& node) {
            os << "nabla::ComputationNode[name: " << node.get_name()
               << ", local_grad: [" << node.get_local_grad().first << ", " << node.get_local_grad().second
               << "], dependencies_indices: [" << node.get_tensor_dependencies().first
               << ", " << node.get_tensor_dependencies().second << "]]";
            return os;
        }

        static inline unsigned int node_id_generator = 0;

    private:
        std::string m_node_name;
        std::pair<double, double> m_local_grad;
        std::pair<node_index_t, node_index_t> m_tensor_indices;
    };

    struct GradientTape {
        GradientTape(const GradientTape&) = delete;

        static GradientTape& instance() {
            static GradientTape s_instance;
            return s_instance;
        }

        static void clean() { instance().tape = std::vector<ComputationNode>(); }

        static void list() {
            std::cout << "nabla::GradientTape[tape:" << std::endl;
            for (const auto& node : instance().tape)
                std::cout << "   " << node << std::endl;
        }

        std::vector<ComputationNode> get_tape() const { return this->tape; }
        const ComputationNode& get_computation_node(node_index_t index) const {
            return this->tape.at(index);
        }

        node_index_t push_node(
            const std::string& node_name,
            const std::pair<double, double>& local_grad,
            const std::pair<node_index_t, node_index_t>& tensor_indices);
        node_index_t push_node(const std::string& node_name, double weight, node_index_t tensor_index);
        node_index_t push_leaf_node(const std::string& node_name);

    private:
        GradientTape() {}
        std::vector<ComputationNode> tape{};
    };
} // namespace nabla

#endif
