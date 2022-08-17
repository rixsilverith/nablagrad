#include "tensor_ops.hpp"
#include "gradient_tape.hpp"

namespace nabla {
    /*
    uint32_t node_is_in_tape(const std::string& tensor_name) {
        for (uint32_t i = 0; i < gradient_tape.size(); ++i)
            if (gradient_tape.at(i).get_name() == tensor_name) return i;
        return -1;
    }*/

    Tensor MultBackward(const Tensor& ltensor, const Tensor& rtensor) {
        double primal = ltensor.get_primal() * rtensor.get_primal();
        Tensor mult_tensor{primal, false};

        // record MultBackward operation on the gradient tape. This is done by pushing
        // a ComputationNode 
        node_index_t tensor_index = GradientTape::instance().push_node(
            "mult_backward_" + std::to_string(ComputationNode::node_id_generator++),
            { rtensor.m_primal, ltensor.m_primal },
            { ltensor.node_gradtape_index, rtensor.node_gradtape_index });

        mult_tensor.node_gradtape_index = tensor_index;
        std::cout << "Added tensor " << mult_tensor << " to computation graph" << std::endl;

        return mult_tensor;

        /*
        // check if the ltensor operand is already in the gradient tape
        uint32_t ltensor_index = node_is_in_tape(ltensor.m_name);
        if (ltensor_index == -1) { // ltensor not in the gradient tape
            if (ltensor.m_dependencies.size() == 0) {
                ComputationGraphNode tensor_node(&ltensor, rtensor.m_primal);
                gradient_tape.push_back(std::move(tensor_node));
            }
        }

        if (ltensor.m_dependencies.size() == 0) {
            // ltensor operand is a leaf node in the computation graph
            uint32_t ltensor_index = node_is_in_tape(ltensor.m_name);
            if (ltensor_index == -1) {
                ComputationGraphNode tensor_node(&ltensor, rtensor.m_primal);
                gradient_tape.push_back(std::move(tensor_node));
                mult_tensor.m_dependencies.emplace_back(ltensor.m_name);
            } 
        } else {
            // ltensor operand is not a leaf node, but a temporary tensor which must be
            // moved into the gradient tape inside its corresponding computation node
            uint32_t ltensor_index = node_is_in_tape(ltensor.m_name);
            if (ltensor_index == -1) {
                ComputationGraphNode tensor_node(std::move(ltensor), rtensor.m_primal);
                gradient_tape.push_back(std::move(tensor_node));
                mult_tensor.m_dependencies.emplace_back(ltensor.m_name);
            }
        }

     
        return mult_tensor;*/
    }
} // namespace nabla