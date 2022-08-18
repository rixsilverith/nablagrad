[![License](https://img.shields.io/github/license/rixsilverith/nablagrad?color=g)](https://mit-license.org/)
![Cpp Version](https://img.shields.io/badge/C%2B%2B-17+-green)

# ∇grad (nablagrad) automatic differentiation engine

Automatic differentiation is a technique to automatically perform efficient and analytically precise partial 
differentiation on a given mathematical function or expression.

*nablagrad* is yet another automatic differentiation engine supporting both forward and reverse-mode autodiff.

---

## Overview

Automatic differentiation has two different modes of operation known as *forward accumulation mode* (or
*tangent linear mode*) and *reverse accumulation mode* (or *cotangent linear mode*). 

*nablagrad* makes use of [dual numbers](https://en.wikipedia.org/wiki/Dual_number), together with operator 
overloading, to perform partial differentiation in forward-mode. In reverse-mode, a gradient tape (known as 
Wengert list) is used to represent an abstract computation graph that is built progressively when evaluating 
the given mathematical expression. Upon completion, the graph is traversed backwards, computing and propagating 
the corresponding partial derivatives using the chain rule.

For a complete, more detailed explanation of how these modes work see 
[Baydin et al. (2018)](https://arxiv.org/abs/1502.05767).

---

## Some usage examples

The full implementation of the following examples (and more) can be found in the [examples](examples) directory. 
Note that in these examples we use template functions to be able to compare *nablagrad* computations with 
`nabla::Dual` and `nabla::Tensor` to the finite differences computation.

### Reverse-mode gradient computation

Let $f(x_0, x_1) = \ln(x_0) + x_0x_1 + \sin(x_1)$. Implement $f(x_0, x_1)$ as a template function to allow 
computing both the value of the function and the gradient.

```cpp
template<typename T> T f(const std::vector<T>& x) {
    return log(x.at(0)) + x.at(0) * x.at(1) + sin(x.at(1));
}
```

Using the [`nabla::Tensor`](nablagrad/tensor.hpp) structure and the [`nabla::grad()`](nablagrad/core.hpp) 
function the gradient $\nabla f(x_0, x_1)$ can be easily computed and evaluated at some vector $x$ as

```cpp
std::vector<double> x = { 2.0, -3.0 };

auto grad_f = nabla::grad(f<nabla::Tensor>);
std::cout << "∇f = " << grad_f(x) << std::endl;
```

Gradient computation can also be performed (less efficiently, see [Baydin et al. (2018)](https://arxiv.org/abs/1502.05767)) 
with forward-mode by using [`nabla::Dual`](nablagrad/dual.hpp) and [`nabla::grad_forward()`](nablagrad/core.hpp) 
instead of `nabla::Tensor` and `nabla::grad()`.

### Forward-mode partial differentiation

Consider a function $f(x_0, x_1) = x_0^2x_1 + x_1$. Partial differentiation (for instance, $\partial f/\partial x_0$) 
can be performed by declaring all variables as `nabla::Dual` and setting the adjoint to $1.0$ of the variable to be 
differentiated with respect to.

```cpp
template <typename T> f(const std::vector<T>& x) {
    return pow(x.at(0), 2) * x.at(1) + x.at(1);
}
```

```cpp
nabla::Dual x_0{-4.3, 1.0};
nabla::Dual x_1{6.2};
std::vector<nabla::Dual> x = { x_0,  x_1 };

nabla::Dual df_x_0 = f<nabla::Dual>(x);
std::cout << "∂f/∂x_0 = " << x_0.get_adjoint() << std::endl;
```

---

## Installation

By default, *nablagrad* is installed in the `/usr/include` directory by running
```bash
git clone https://github.com/rixsilverith/nablagrad.git && cd nablagrad && make install
```

However, the installation directory can be modified according to the `INSTALL_DIR` variable in the [Makefile](Makefile).
Once installed, *nablagrad* should be available as the [`<nablagrad/nabla.h>`](nablagrad/nabla.h) header file.

> **Note** As of now, only Linux supports this kind of installation. For other systems, local installation
must be used instead.

### Local installation

*nablagrad* can also be installed locally by copying the `nablagrad` folder to the desired location. Then *nablagrad*
should be available by importing the header file [`nablagrad/nabla.h`](nablagrad/nabla.h).

### Requirements

A compiler supporting, at least, C++17 is needed in order to compile *nablagrad* from source.

---

## About the design

Currently, reverse-mode uses a gradient tape to record the operations performed between tensors, but not the tensors
themselves. The downside of this approach is that the gradients cannot be propagated to the tensor; i.e. gradients
are not stored in the tensors, but in an allocated output vector. Therefore, is not possible (as it is in forward-mode) to
get the adjoints (gradients) for each tensor directly from it via the `get_adjoint()` method. Instead, it is needed
the access the corresponding partial derivative in the gradient vector by its index.

---

## License

*nablagrad* is licensed under the MIT License. See [LICENSE](LICENSE) for more information. A copy of the license can be found along with the code.

---

## References

- Baydin, A. G.; Pearlmutter, B. A.; Radul, A. A.; Siskind, J. M. (2018). *Automatic differentiation in machine learning: a survey*. [https://arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767)
- *Automatic differentiation*. [Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation).
