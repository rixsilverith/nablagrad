#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <string>

#include "gradient_tape.hpp"

namespace nabla {
    struct Tensor {
        static inline const bool NO_GRAD = false;

        Tensor(const Tensor& tensor) : m_name{tensor.m_name}, m_primal{tensor.m_primal} {
            std::cout << "! copying tensor " << tensor << std::endl;
        }

        Tensor(Tensor&& tensor) noexcept : m_name{tensor.m_name} {
            std::cout << "! moved tensor" << std::endl;
            this->m_primal = tensor.m_primal;
            this->node_gradtape_index = tensor.node_gradtape_index;
        }

        Tensor(double primal, bool trace_grad = true) : m_primal{primal} {
            this->m_name = "tensor_" + std::to_string(tensor_id_counter++);
            if (trace_grad) {
                node_index_t idx = GradientTape::instance().push_leaf_node(this->m_name);
                this->node_gradtape_index = idx;
            }
            std::cout << "Init tensor " << *this << std::endl;
        }

        ~Tensor() {
            std::cout << "Destroyed " << *this << std::endl;
        }

        double get_primal() const { return this->m_primal; }
        void set_primal(double primal) { this->m_primal = primal; }

        const std::string& get_name() const { return this->m_name; }
        void set_name(const std::string& name) { this->m_name = name; }

        std::vector<double> backward() const;
        void backward(double adjoint) const;

        friend Tensor operator+(const Tensor& ltensor, const Tensor& rtensor);
        friend Tensor operator-(const Tensor& ltensor, const Tensor& rtensor);
        friend Tensor operator*(const Tensor& ltensor, const Tensor& rtensor);
        friend Tensor operator/(const Tensor& ltensor, const Tensor& rtensor);

        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

        friend Tensor AddBackward(const Tensor& ltensor, const Tensor& rtensor);
        friend Tensor SubBackward(const Tensor& ltensor, const Tensor& rtensor);
        friend Tensor MultBackward(const Tensor& ltensor, const Tensor& rtensor);
        friend Tensor DivBackward(const Tensor& ltensor, const Tensor& rtensor);

        friend Tensor ExpBackward(const Tensor& tensor);
        friend Tensor LogBackward(const Tensor& tensor);
        friend Tensor PowerBackward(const Tensor& tensor, unsigned int pow);

        friend Tensor SinBackward(const Tensor& tensor);
        friend Tensor CosBackward(const Tensor& tensor);

        node_index_t node_gradtape_index = -1; // index of the corresponding computation node in the gradient tape
    private:
        static inline int tensor_id_counter = 0;

        std::string m_name;
        double m_primal;
    };
} // namespace nabla

#endif
