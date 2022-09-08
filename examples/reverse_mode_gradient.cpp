#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>

#include <nablagrad/nabla.h>

// f(x1, x2) = ln(x1) + x1 * x2 - sin(x2)
template<typename T> T f(const std::vector<T>& x) {
    return log(x.at(0)) + x.at(0) * x.at(1) - sin(x.at(1));
}

int main() {
    std::vector<double> x = {2.0, 5.0}; // vector at which the gradient is evaluated
    auto grad_f = nabla::grad(f<nabla::Tensor>);

    std::cout << "x = " << x << std::endl;
    std::cout << "f(x) = " << f<double>(x) << std::endl;
    std::cout << std::setprecision(20) << "∇f(x) = " << grad_f(x) << std::endl;

    nabla::GradientTape::clean(); // Optionally, clean gradient tape

    return EXIT_SUCCESS;
}
