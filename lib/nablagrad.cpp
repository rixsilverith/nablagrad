#include "nablagrad.hpp"
#include <algorithm>

namespace nabla {

std::function<double(double)> derivative(std::function<Dual(Dual)> f) {
    return [f](double x) -> double { return f(Dual(x, 1)).get_adjoint(); };
}

std::function<std::vector<double>(const std::vector<double>&)>
grad(std::function<Dual(const std::vector<Dual>&)> F) {
    auto Df = [F](const std::vector<double>& x) -> std::vector<double> {
        int n = x.size(); // dim

        std::vector<Dual> xe(x.begin(), x.end());
        std::vector<double> Dfx(n); // gradient result
        
        for (size_t i = 0; i < n; ++i) {
            xe[i].set_adjoint(1.);
            Dfx[i] = F(xe).get_adjoint();
            xe[i].set_adjoint(0.);
        }

        return Dfx;
    };
    return Df;
}

std::function<double(const std::vector<double>&, const std::vector<double>&)>
grad_dir(std::function<Dual(const std::vector<Dual>&)> F) {
    auto Df = [F](const std::vector<double>& x, const std::vector<double>& y) -> double {
        std::vector<Dual> z(x.size()); // x and y should have the same size!
        std::transform(x.begin(), x.end(), y.begin(), z.begin(), [](double xi, double yi) {
            return Dual(xi, yi);
        });
        return F(z).get_adjoint();
    };
    return Df;
}

}


