#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>

#include <nablagrad/nabla.h>

template<typename T> T l2norm(const std::vector<T>& x) {
    T t{0};
    for (T xi : x) t += xi * xi;
    return sqrt(t);
}

int main() {
    nabla::Dual x0{-4.3, 1.0}; // set dual adjoint to 1.0 to differentiate wrt x0
    nabla::Dual x1{6.2};
    nabla::Dual x2{-5.8};
    std::vector<nabla::Dual> x = {x0, x1, x2};

    nabla::Dual dl2norm_x0 = l2norm<nabla::Dual>(x);

    std::cout << "∂l2norm/∂x0 = " << dl2norm_x0.get_adjoint() << std::endl;

    return EXIT_SUCCESS;
}
