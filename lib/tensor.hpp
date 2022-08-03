#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <string>

namespace nabla {
    struct Tensor;

    // TensorDependencyNode represents the dependency of a tensor in the reverse-mode computational
    // graph to another tensor. This class act as as wrapper for both the referenced tensor and the 
    // local adjoint that is stored during forward propagation to be later be propagated to the
    // necessary nodes during the backward pass.
    struct TensorDependencyNode {
        TensorDependencyNode(const Tensor *tensor, double adjoint) : m_tensor{tensor}, m_adjoint{adjoint} {}

        const Tensor *get_tensor() const noexcept { return this->m_tensor; }
        double get_adjoint() const noexcept { return this->m_adjoint; }

    private:
        const Tensor* const m_tensor;
        double m_adjoint;
    };

    struct Tensor {
        Tensor() : m_primal{0.}, m_adjoint{0.} {
            this->m_name = "tensor_" + std::to_string(tensor_id_counter++);
            std::cout << "Init tensor " << *this << std::endl;
        }

        Tensor(const Tensor& tensor) : m_name{tensor.m_name}, m_primal{tensor.m_primal}, m_adjoint{tensor.m_adjoint} {
            std::cout << "! copying tensor " << tensor << std::endl;
            //this->m_tensor_dependencies = tensor.m_tensor_dependencies;
        }

        Tensor(double primal) : m_primal{primal}, m_adjoint{0.} {
            this->m_name = "tensor_" + std::to_string(tensor_id_counter++);
            std::cout << "Init tensor " << *this << std::endl;
        }

        Tensor(double primal, double adjoint) : m_primal{primal}, m_adjoint{adjoint} {
            this->m_name = "tensor_" + std::to_string(tensor_id_counter++);
            std::cout << "Init tensor " << *this << std::endl;
        }

        double get_primal() const { return this->m_primal; }
        void set_primal(double primal) { this->m_primal = primal; } 

        double get_adjoint() const { return this->m_adjoint; }
        void set_adjoint(double adjoint) { this->m_adjoint = adjoint; }

        const std::string& get_name() const { return this->m_name; }
        void set_name(const std::string& name) { this->m_name = name; }

        void backward() const;
        void backward(double adjoint) const;

        friend Tensor operator+(const Tensor& ltensor, const Tensor& rtensor);
        friend Tensor operator-(const Tensor& ltensor, const Tensor& rtensor);
        friend Tensor operator*(const Tensor& ltensor, const Tensor& rtensor);
        //friend Dual operator/(const Tensor& ltensor, const Tensor& rtensor);

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    private:
        static inline int tensor_id_counter = -1;

        std::string m_name;
        double m_primal;
        // NB: So, making m_adjoint mutable may not seem the best approach to update the adjoint
        // value during the backward pass, but imho it is the cleanest way without messing stuff
        // up and without triggering a bunch of error messages from our friend the compiler. Thus
        // this is it for now. A better implementation may work out in the future (hopefully).
        mutable double m_adjoint;
        std::vector<TensorDependencyNode> m_tensor_dependencies;
    };



    /*
    struct TensorAddBackward : public Tensor {
        TensorAddBackward(const Tensor& ltensor, const Tensor& rtensor);
    };

    struct TensorSubBackward : public Tensor {
        TensorSubBackward(const Tensor& ltensor, const Tensor& rtensor);
    };

    struct TensorMultBackward : public Tensor {
        TensorMultBackward(const Tensor& ltensor, const Tensor& rtensor);
    };*/

    /*
    struct TensorDivBackward : public Tensor {
        TensorDivBackward(const Tensor& ltensor, const Tensor& rtensor);
    };*/
} // namespace nabla

#endif
