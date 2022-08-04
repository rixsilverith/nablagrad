#include "dual.hpp"

namespace nabla {
    const Dual& Dual::operator=(double x) {
        this->m_primal = x;
        this->m_adjoint = 0.;
        return *this;
    }

    const Dual& Dual::operator=(const Dual& d) {
        if (this != &d) {
            this->m_primal = d.get_primal();
            this->m_adjoint = d.get_adjoint();
        }
        return *this;
    }

    const Dual& Dual::operator+=(const Dual& d) {
        this->m_primal += d.get_primal();
        this->m_adjoint += d.get_adjoint();
        return *this;
    }

    Dual Dual::operator+(const Dual& d) const {
        return Dual(this->m_primal + d.get_primal(), this->m_adjoint + d.get_adjoint());
    }

    Dual Dual::operator-(const Dual& d) const {
        return Dual(this->m_primal - d.get_primal(), this->m_adjoint - d.get_adjoint());
    }

    Dual Dual::operator*(const Dual& d) const {
        return Dual(this->m_primal * d.get_primal(), 
            this->m_adjoint * d.get_primal() + this->m_primal * d.get_adjoint());
    }

    Dual Dual::operator/(const Dual& d) const {
        return Dual(this->m_primal / d.get_primal(), 
            this->m_adjoint / d.get_primal() - this->m_primal * d.get_adjoint() / (d.get_primal() * d.get_primal()));
    }
}
