#include "duals.hpp"

namespace nabla {
    Dual sin(Dual d) { return Dual(::sin(d.m_primal), d.m_adjoint * ::cos(d.m_primal)); }
    Dual cos(Dual d) { return Dual(::cos(d.m_primal), -d.m_adjoint * ::sin(d.m_primal)); }
    Dual exp(Dual d) { return Dual(::exp(d.m_primal), d.m_adjoint * ::exp(d.m_primal)); }
    Dual log(Dual d) { return Dual(::log(d.m_primal), d.m_adjoint / d.m_primal); }

    Dual abs(Dual d) {
        int sign = d.m_primal == 0 ? 0 : d.m_primal / ::abs(d.m_primal);
        return Dual(::abs(d.m_primal), d.m_adjoint * sign);
    }

    Dual power(Dual d, double p) { return Dual(::pow(d.m_primal, p), p * d.m_adjoint * ::pow(d.m_primal, p - 1)); }

    Dual sqrt(const Dual& d) {
        double sq = ::sqrt(d.m_primal);
        return Dual(sq, 0.5 * d.m_adjoint / sq);
    }
}
