#include "tensor_ops.hpp"
#include "gradient_tape.hpp"

#include <cmath>

namespace nabla {
    Tensor AddBackward(const Tensor& ltensor, const Tensor& rtensor) {
        double primal = ltensor.m_primal + rtensor.m_primal;
        Tensor add_tensor{primal, false};

        node_index_t tensor_index = GradientTape::instance().push_node(
            "add_backward_" + std::to_string(ComputationNode::node_id_generator++),
            { 1.0, 1.0 }, { ltensor.node_gradtape_index, rtensor.node_gradtape_index });

        add_tensor.node_gradtape_index = tensor_index;
        std::cout << "Added tensor " << add_tensor << " to computation graph" << std::endl;
        return add_tensor;
    }

    Tensor SubBackward(const Tensor& ltensor, const Tensor& rtensor) {
        double primal = ltensor.m_primal - rtensor.m_primal;
        Tensor sub_tensor{primal, false};

        node_index_t tensor_index = GradientTape::instance().push_node(
            "sub_backward_" + std::to_string(ComputationNode::node_id_generator++),
            { 1.0, -1.0 }, { ltensor.node_gradtape_index, rtensor.node_gradtape_index });

        sub_tensor.node_gradtape_index = tensor_index;
        std::cout << "Added tensor " << sub_tensor << " to computation graph" << std::endl;
        return sub_tensor;
    }

    Tensor MultBackward(const Tensor& ltensor, const Tensor& rtensor) {
        double primal = ltensor.get_primal() * rtensor.get_primal();
        Tensor mult_tensor{primal, false};

        node_index_t tensor_index = GradientTape::instance().push_node(
            "mult_backward_" + std::to_string(ComputationNode::node_id_generator++),
            { rtensor.m_primal, ltensor.m_primal },
            { ltensor.node_gradtape_index, rtensor.node_gradtape_index });

        mult_tensor.node_gradtape_index = tensor_index;
        std::cout << "Added mult_tensor " << mult_tensor << " to computation graph" << std::endl;
        return mult_tensor;
    }

    Tensor DivBackward(const Tensor& ltensor, const Tensor& rtensor) {
        double primal = ltensor.m_primal / rtensor.m_primal;
        Tensor div_tensor{primal, false};

        div_tensor.node_gradtape_index = GradientTape::instance().push_node(
            "div_backward_" + std::to_string(ComputationNode::node_id_generator++),
            { rtensor.m_primal, ltensor.m_primal },
            { ltensor.node_gradtape_index, rtensor.node_gradtape_index });

        std::cout << "Added tensor " << div_tensor << " to computation graph" << std::endl;
        return div_tensor;
    }

    Tensor ExpBackward(const Tensor& tensor) {
        double primal = ::exp(tensor.m_primal);
        Tensor exp_tensor{primal, false};

        node_index_t tensor_index = GradientTape::instance().push_node(
            "exp_backward_" + std::to_string(ComputationNode::node_id_generator++),
            primal, tensor.node_gradtape_index);

        exp_tensor.node_gradtape_index = tensor_index;
        std::cout << "Added tensor " << exp_tensor << " to computation graph" << std::endl;
        return exp_tensor;
    }

    Tensor LogBackward(const Tensor& tensor) {
        double primal = ::log(tensor.m_primal);
        Tensor log_tensor{primal, false};

        node_index_t tensor_index = GradientTape::instance().push_node(
            "log_backward_" + std::to_string(ComputationNode::node_id_generator++),
            1 / tensor.m_primal, tensor.node_gradtape_index);

        log_tensor.node_gradtape_index = tensor_index;
        std::cout << "Added log_tensor " << log_tensor << " to computation graph" << std::endl;
        return log_tensor;
    }

    Tensor PowerBackward(const Tensor& tensor, unsigned int power) {
        double primal = ::pow(tensor.m_primal, power);
        Tensor power_tensor{primal, false};

        node_index_t tensor_index = GradientTape::instance().push_node(
            "power_backward_" + std::to_string(ComputationNode::node_id_generator++),
            power * ::pow(tensor.m_primal, power - 1),
            tensor.node_gradtape_index);

        power_tensor.node_gradtape_index = tensor_index;
        std::cout << "Added tensor " << power_tensor << " to computation graph" << std::endl;
        return power_tensor;
    }

    Tensor SinBackward(const Tensor& tensor) {
        double primal = ::sin(tensor.m_primal);
        Tensor sin_tensor{primal, false};

        sin_tensor.node_gradtape_index = GradientTape::instance().push_node(
            "sin_backward_" + std::to_string(ComputationNode::node_id_generator++),
            ::cos(tensor.m_primal), tensor.node_gradtape_index);

        std::cout << "Added sin_tensor " << sin_tensor << " to computation graph" << std::endl;
        return sin_tensor;
    }

    Tensor CosBackward(const Tensor& tensor) {
        double primal = ::cos(tensor.m_primal);
        Tensor cos_tensor{primal, false};

        cos_tensor.node_gradtape_index = GradientTape::instance().push_node(
            "cos_backward_" + std::to_string(ComputationNode::node_id_generator++),
            -::sin(tensor.m_primal), tensor.node_gradtape_index);

        std::cout << "Added tensor " << cos_tensor << " to computation graph" << std::endl;
        return cos_tensor;
    }

    Tensor exp(const Tensor& tensor) { return ExpBackward(tensor); }
    Tensor log(const Tensor& tensor) { return LogBackward(tensor); }
    Tensor pow(const Tensor& tensor, unsigned int power) { return PowerBackward(tensor, power); }

    Tensor sin(const Tensor& tensor) { return SinBackward(tensor); }
    Tensor cos(const Tensor& tensor) { return CosBackward(tensor); }
} // namespace nabla
