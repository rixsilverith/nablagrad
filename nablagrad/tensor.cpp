#include "tensor.hpp"
#include "tensor_ops.hpp"
#include "core.hpp"

#include <iostream>

namespace nabla {
    std::vector<double> Tensor::backward() const {
        //std::cout << "[nabla::Tensor::backward] called on tensor " << *this << std::endl;

        node_index_t size = GradientTape::instance().get_tape().size();
        std::vector<double> gradients(size, 0);
        gradients.at(this->node_gradtape_index) = 1.0;

        //std::cout << "grad: ";
        //for (auto i : gradients) { std::cout << i << ", "; }
        //std::cout << std::endl;
        //std::cout << gradients << std::endl;

        //std::cout << size << std::endl;

        for (node_index_t i = size - 1; i >= 0; --i) {
            const ComputationNode& node = GradientTape::instance().get_computation_node(i);

            //std::cout << "propagating from node " << node << std::endl;
            //std::cout << "tape index: " << i << std::endl;
            // weight = localgrad for node * adjoint of the previous node
            double weight = node.get_local_grad().first * gradients.at(i);
            gradients.at(node.get_tensor_dependencies().first) += weight;

            weight = node.get_local_grad().second * gradients.at(i);
            gradients.at(node.get_tensor_dependencies().second) += weight;
        }

        //std::cout << "resulting gradient: " << gradients << std::endl;
        return gradients;
    }

    /*
    void Tensor::backward() const {
        this->m_adjoint = 1.;

        std::cout << "[nabla::Tensor::backward] Called backward on tensor " << *this << std::endl;
        std::cout << "      which depends on tensors:" << std::endl;

        int tensor_dep_index = 0;
        for (auto& tensor_dep : this->m_dependencies) {
            if (tensor_dep_index++ == 0) continue;
            std::cout << "       └── " << *(tensor_dep.get_tensor()) << " with dependency adjoint " << tensor_dep.get_adjoint() << std::endl;
        }

        tensor_dep_index = 0;
        for (auto& tensor_dep : this->m_dependencies) {
            if (tensor_dep_index++ == 0) continue;
            double adjoint = this->m_adjoint * tensor_dep.get_adjoint();

            std::cout << "[nabla::Tensor::backward] Propagating adjoint " << adjoint 
                      << " to tensor " << tensor_dep.get_tensor()->get_name() << std::endl;
            tensor_dep.get_tensor()->backward(adjoint);
        }
    }*/

    void Tensor::backward(double adjoint) const {
        double adj = adjoint;

        std::cout << "[nabla::Tensor::backward] Tensor " << *this << " received adjoint " << adjoint << std::endl;;
        std::cout << "      depends on tensors:" << std::endl;

        int tensor_dep_index = 0;
        for (auto& tensor_dep : this->m_dependencies) {
            if (tensor_dep_index++ == 0) continue;
            std::cout << "       └── " << *(tensor_dep.get_tensor()) 
                      << " with dependency adjoint " << tensor_dep.get_adjoint() << std::endl;
        }

        tensor_dep_index = 0;
        for (auto& tensor_dep : this->m_dependencies) {
            if (tensor_dep_index++ == 0) continue;
            double local_grad = adjoint * tensor_dep.get_adjoint();

            std::cout << "[nabla::Tensor::backward] Propagating adjoint " << local_grad
                      << " to tensor " << tensor_dep.get_tensor()->get_name() << std::endl;

            tensor_dep.get_tensor()->backward(local_grad);
            adj += local_grad;
        }
        
        this->m_adjoint += adj;
    }

    Tensor operator*(const Tensor& ltensor, const Tensor& rtensor) {
        return MultBackward(ltensor, rtensor);
    }

    Tensor operator+(const Tensor& ltensor, const Tensor& rtensor) {
        double primal = ltensor.m_primal + rtensor.m_primal;
        Tensor add_tensor{primal};

        add_tensor.m_dependencies.emplace_back(TensorDependencyNode(&ltensor, 1.));
        add_tensor.m_dependencies.emplace_back(TensorDependencyNode(&ltensor, 1.));
        add_tensor.m_dependencies.emplace_back(TensorDependencyNode(&rtensor, 1.));

        int tensor_dep_index = 0;
        std::cout << "Added add_tensor " << add_tensor << " to backward computational graph" << std::endl;
        std::cout << "      depends on tensors:" << std::endl;
        for (auto& tensor_dep : add_tensor.m_dependencies) {
            if (tensor_dep_index++ == 0) continue;
            std::cout << "       └── " << *(tensor_dep.get_tensor()) 
                      << " with dependency adjoint " << tensor_dep.get_adjoint() << std::endl; 
        }

        return add_tensor;
    }

    Tensor operator-(const Tensor& ltensor, const Tensor& rtensor) {
        double primal = ltensor.m_primal - rtensor.m_primal;
        Tensor sub_tensor{primal};

        sub_tensor.m_dependencies.emplace_back(TensorDependencyNode(&ltensor, 1.));
        sub_tensor.m_dependencies.emplace_back(TensorDependencyNode(&ltensor, 1.));
        sub_tensor.m_dependencies.emplace_back(TensorDependencyNode(&rtensor, -1.));

        int tensor_dep_index = 0;
        std::cout << "Added sub_tensor " << sub_tensor << " to backward computational graph" << std::endl;
        std::cout << "      depends on tensors:" << std::endl;
        for (auto& tensor_dep : sub_tensor.m_dependencies) {
            if (tensor_dep_index++ == 0) continue;
            std::cout << "       └── " << *(tensor_dep.get_tensor()) 
                      << " with dependency adjoint " << tensor_dep.get_adjoint() << std::endl; 
        }

        return sub_tensor;
    }

    /*
    Tensor operator*(const Tensor& ltensor, const Tensor& rtensor) {
        double primal = ltensor.m_primal * rtensor.m_primal;
        Tensor mult_tensor{primal};

        mult_tensor.m_dependencies.emplace_back(TensorDependencyNode(&ltensor, rtensor.m_primal));
        mult_tensor.m_dependencies.emplace_back(TensorDependencyNode(&ltensor, rtensor.m_primal));
        mult_tensor.m_dependencies.emplace_back(TensorDependencyNode(&rtensor, ltensor.m_primal));

        int tensor_dep_index = 0;
        std::cout << "Added mult_tensor " << mult_tensor << " to backward computational graph" << std::endl;
        std::cout << "      depends on tensors:" << std::endl;
        for (auto& tensor_dep : mult_tensor.m_dependencies) {
            if (tensor_dep_index++ == 0) continue;
            std::cout << "       └── " << *(tensor_dep.get_tensor()) 
                      << " with dependency adjoint " << tensor_dep.get_adjoint() << std::endl; 
        }

        return mult_tensor;
    }*/
        
    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "nabla::Tensor[name: " << tensor.m_name << ", primal: " << tensor.m_primal
           << ", grad_tape_node_index: " << tensor.node_gradtape_index << "]";
        return os;
    }

    /*
    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "nabla::Tensor[name: " << tensor.m_name << ", primal: " << tensor.m_primal << ", adjoint: " 
        << tensor.m_adjoint << ", node_index: " << tensor.node_gradtape_index << "]";
        return os;
    }*/

    /*
    Tensor operator+(const Tensor& ltensor, const Tensor& rtensor) {
        return TensorDivBackward(ltensor, rtensor);
    }*/

    /*
    TensorAddBackward::TensorAddBackward(Tensor *ltensor, Tensor *rtensor) {
        //std::cout << "tensor_id_counter on add tensor back: " << tensor_id_counter << std::endl;
        //int assigned_id = tensor_id_counter++;
        //this->m_name = "v_" + std::to_string(assigned_id);

        this->m_primal = ltensor->get_primal() + rtensor->get_primal();
        //std::cout << "[nabla::TensorAddBackward] new with primal = " << this->m_primal << std::endl;

        std::cout << "Added nabla::TensorAddBackward[name: " << this->m_name << ", primal: " 
                  << this->m_primal << ", adjoint: " << this->m_adjoint << "]" << std::endl;
        std::cout << "      depends on tensors (adjoints not updated):" << std::endl;
        std::cout << "       └── " << *ltensor;
        std::cout << "       └── " << *rtensor;

        this->m_dependencies.push_back(TensorDependencyNode(ltensor, 1.));
        this->m_dependencies.push_back(TensorDependencyNode(rtensor, 1.));
    }

    TensorSubBackward::TensorSubBackward(const Tensor& ltensor, const Tensor& rtensor) {
            int assigned_id = tensor_id_counter++;
            this->m_name = "v_" + std::to_string(assigned_id);
        this->m_primal = ltensor.get_primal() - rtensor.get_primal();
        std::cout << "[nabla::TensorSubBackward] new with primal = " << this->m_primal << std::endl;
        this->m_dependencies.emplace_back(TensorDependencyNode(ltensor, 1.));
        this->m_dependencies.emplace_back(TensorDependencyNode(rtensor, -1.));
    }

    TensorMultBackward::TensorMultBackward(Tensor *ltensor, Tensor *rtensor) {
        //std::cout << "instantiated tensor mult back with id: " << tensor_id_counter << std::endl;
        //int assigned_id = tensor_id_counter++;
        //this->m_name = "v_" + std::to_string(assigned_id);

        this->m_primal = ltensor->get_primal() * rtensor->get_primal();

        std::cout << "Added nabla::TensorMultBackward[name: " << this->m_name << ", primal: " 
                  << this->m_primal << ", adjoint: " << this->m_adjoint << "]" << std::endl;
        std::cout << "      depends on tensors (adjoints not updated):" << std::endl;
        std::cout << "       └── " << *ltensor;
        std::cout << "       └── " << *rtensor;

        //this->deps.push_back(std::ref(ltensor));

        this->m_dependencies.emplace_back(TensorDependencyNode(ltensor, rtensor->get_primal()));
        this->m_dependencies.emplace_back(TensorDependencyNode(rtensor, ltensor->get_primal()));
    }*/

    
} // namespace nabla
