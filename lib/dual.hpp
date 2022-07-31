#ifndef DUAL_NUMBER_H
#define DUAL_NUMBER_H

#include <iostream>
#include <cmath>

namespace nabla {
    // Representation of a dual number. Dual numbers are of the form `x + ye`, where 
    // e^2 = 0 and e != 0. In this implementation we refer to `x` and `y` has the `primal` and
    // `adjoint` attributes, respectively. Sometimes the `adjoint` is called the `tangent`.
    //
    struct Dual {
        constexpr Dual() : m_primal{0.}, m_adjoint{0.} {}
        constexpr Dual(const Dual& d) : m_primal{d.get_primal()}, m_adjoint{d.get_adjoint()} {}
        constexpr Dual(double primal) : m_primal{primal}, m_adjoint{0.} {}
        constexpr Dual(double primal, double adjoint) : m_primal{primal}, m_adjoint{adjoint} {}

        constexpr double get_primal() const { return this->m_primal; }
        void set_primal(double val) { this->m_primal = val; }

        constexpr double get_adjoint() const { return this->m_adjoint; }
        void set_adjoint(double val) { this->m_adjoint = val; }

        const Dual& operator=(const Dual& d);
        const Dual& operator=(double x);
        const Dual& operator+=(const Dual& d);

        Dual operator+(const Dual& d) const;
        Dual operator-(const Dual& d) const;
        Dual operator*(const Dual& d) const;
        Dual operator/(const Dual& d) const;

        friend Dual sin(Dual d);
        friend Dual cos(Dual d);
        friend Dual exp(Dual d);
        friend Dual log(Dual d);
        friend Dual abs(Dual d);
        friend Dual power(Dual d, double pow);
        friend Dual sqrt(const Dual& d);

    private:
        double m_primal;
        double m_adjoint;
    };
}

#endif
