#include "core.hpp"
#include <algorithm>

namespace nabla {
    // reverse-mode gradient computation
    std::function<Gradient(const RealVec&)> grad(std::function<Tensor(const TensorVec&)> f) {
        auto Df = [f](const RealVec& x) -> Gradient {
            TensorVec input_tensor(x.begin(), x.end());
            Gradient full_gradient = f(input_tensor).backward();

            size_t n = x.size();
            Gradient grad_vec(n);
            for (size_t i = 0; i < n; ++i) {
                node_index_t tensor_index = input_tensor.at(i).node_gradtape_index;
                grad_vec.at(i) = full_gradient.at(tensor_index);
            }
            return grad_vec;
        };
        return Df;
    }

    // derivative computation using forward-mode
    std::function<double(double)> grad_forward(std::function<Dual(Dual)> f) {
        return [f](double x) -> double { return f(Dual(x, 1)).get_adjoint(); };
    }

    // gradient computation using forward-mode
    std::function<RealVec(const RealVec&)> grad_forward(std::function<Dual(const DualVec&)> F) {
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

    // compute and evaluate gradient on given vector
    RealVec grad_forward(std::function<Dual(const DualVec&)> f, const RealVec& x) {
        auto gradient = grad_forward(f);
        return gradient(x);
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
