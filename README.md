[![License](https://img.shields.io/github/license/rixsilverith/nablagrad?color=g)](https://mit-license.org/)
![Cpp Version](https://img.shields.io/badge/C%2B%2B-17+-green)

# ∇grad (nablagrad)

Automatic differentiation (also known as *autodiff*) is a technique to automatically compute and evaluate partial (or ordinary) 
derivatives of a given mathematical function.

*nablagrad* is yet another automatic differentiation library written in C++. Despite its main purpose being
learning how autodiff works and how it may be implemented, it should be a fast and reliable tool to perform first-order
differentiation in both forward and reverse accumulation mode.

---

## Overview

Automatic differentiation has two different modes of operation known as *forward accumulation mode* (or
*tangent linear mode*) and *reverse accumulation mode* (or *cotangent linear mode*). *nablagrad* makes use of the 
[`nabla::Dual`](nablagrad/dual.hpp) structure representing a [dual number](https://en.wikipedia.org/wiki/Dual_number),
together with operator overloading, to perform partial differentiation in forward-mode.

In the case of reverse-mode, *nablagrad* uses the [`nabla::Tensor`](nablagrad/tensor.hpp) structure (which is an extended 
and adapted version of `nabla::Dual`) to progressively build a computational graph from the given mathematical expression, 
which upon completion is traversed backwards to compute and propagate partial derivatives using the chain rule.

For a complete, more detailed explanation of how these modes work see [Baydin et. al. (2018)](https://arxiv.org/abs/1502.05767).

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

In order to compile *nablagrad* from source, a compiler that supports at least C++17 is needed.

---

## Some usage examples

The full implementation of the following examples (and more) can be found in the [examples](examples) directory. Note that 
in these examples we use template functions to be able to compare *nablagrad* computations with `nabla::Dual` and `nabla::Tensor`
to the finite differences computation.

### Forward-mode partial differentiation

Consider a function $f(x_0, x_1) = x_0^2x_1 + x_1$. Partial differentiation (for instance, $\partial f/\partial x_0$) 
can be performed by declaring all variables as `nabla::Dual` and setting the adjoint to $1.0$ of the variable to be differentiated 
with respect to.

```cpp
template <typename T> f(const std::vector<T>& x) {
    return pow(x.at(0)) * x.at(1) + x.at(1);
}
```

```cpp
nabla::Dual x_0{-4.3, 1.0};
nabla::Dual x_1{6.2};
std::vector<nabla::Dual> x = { x_0,  x_1 };
std::vector<double> x_f = { -4.3, 6.2 };

nabla::Dual df_x0 = f<nabla::Dual>(x);
double finite_diff = (f<double>(x_f + 0.001) - f<double>(x_f - 0.001)) / 0.0002;

std::cout << "∂f/∂x_0 = " << x_0.get_adjoint() << std::endl;
std::cout << "finite differences: ∂f/∂x_0 = " << finite_diff << std::endl;
```

### Reverse-mode gradient computation

In this example we compute the gradient of $f(x_0, x_1) = \ln(x_0) + x_0x_1 + \sin(x_1)$ using reverse-mode.
Define $f(x_0, x_1)$ as a template function to allow computing both the value of the function and the gradient.

```cpp
template <typename T> T f(const std::vector<T>& x) {
    return log(x.at(0)) + x.at(0) * x.at(1) + sin(x.at(1));
}
```

Using the `nabla::Tensor` structure and the `nabla::grad()` function the gradient can be computed and evaluated
at the vector $x$ easily as

```cpp
std::vector<double> x = { 2.0, -3.0 };

auto grad_f = nabla::grad(f<nabla::Tensor>);
double finite_diff = (f<double>(x + 0.001) - f<double>(x - 0.001)) / 0.0002;

std::cout << "autodiff: " << grad_f(x) << std::endl;
std::cout << "finite differences: " << finite_diff << std::endl;

```

Gradient computation can also be performed (less efficiently, see [Baydin et. al. (2018)](https://arxiv.org/abs/1502.05767)) 
with forward-mode by using `nabla::Dual` and `nabla::grad_forward()` instead of `nabla::Tensor` and `nabla::grad()`.

---

## License

*nablagrad* is licensed under the MIT License. See [LICENSE](LICENSE) for more information. A copy of the license can be found along with the code.

---

## References

- Baydin, A. G.; Pearlmutter, B. A.; Radul, A. A.; Siskind, J. M. (2018). *Automatic differentiation in machine learning: a survey*. [https://arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767)
- *Automatic differentiation*. [Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation).
