#include "tensor_ops.hpp"
#include "gradient_tape.hpp"

namespace nabla {
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
    }
} // namespace nabla
