#include "nablagrad.hpp"
#include <algorithm>

namespace nabla {
    // derivative computation using forward-mode
    std::function<double(double)> derivative(std::function<Dual(Dual)> f) {
        return [f](double x) -> double { return f(Dual(x, 1)).get_adjoint(); };
    }

    // gradient computation using forward-mode
    std::function<RealVec(const RealVec&)> grad(std::function<Dual(const DualVec&)> F) {
        auto Df = [F](const RealVec& x) -> RealVec {
            size_t n = x.size(); // dimension

            DualVec dual(x.begin(), x.end()); // transform x into a dual vector
            RealVec gradient(n);
        
            for (size_t i = 0; i < n; ++i) {
                dual[i].set_adjoint(1.); // enable differentiation wrt the i-th variable
                gradient[i] = F(dual).get_adjoint(); // evaluate F using the dual vector and get the adjoint
                dual[i].set_adjoint(0.); // disable diff wrt the i-th variable for next iteration
            }
            return gradient;
        };
        return Df;
    }

    // directional derivative computation using forward-mode
    std::function<double(const RealVec&, const RealVec&)> grad_dir(std::function<Dual(const DualVec&)> F) {
        auto Df = [F](const RealVec& x, const RealVec& y) -> double {
            std::vector<Dual> z(x.size()); // x and y should have the same size!
            std::transform(x.begin(), x.end(), y.begin(), z.begin(), [](double xi, double yi) {
                return Dual(xi, yi);
            });
            return F(z).get_adjoint();
        };
        return Df;
    }
}
