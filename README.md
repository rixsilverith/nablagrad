[![License](https://img.shields.io/github/license/rixsilverith/nablagrad?color=g)](https://mit-license.org/)
![Cpp Version](https://img.shields.io/badge/C%2B%2B-17+-green)

# ∇grad (nablagrad) automatic differentiation engine

Automatic differentiation is a technique to automatically perform efficient and analytically precise partial
differentiation on a given mathematical function or expression.

*nablagrad* is yet another automatic differentiation engine supporting both forward and reverse-mode autodiff.
Examples can be found in the [examples](examples) directory.

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

## Some quick usage examples

The full implementation of the following examples can be found in the [examples](examples) directory.
Note that in these examples we use template functions to be able to compare *nablagrad* computations with
`nabla::Dual` and `nabla::Tensor` to the finite differences computation.

Before running these examples, see [Installation and usage](#installation-and-usage).
Examples can be compiled with `make examples`.

### Reverse-mode gradient computation

Let $f(x_0, x_1) = \ln(x_0) + x_0x_1 - \sin(x_1)$. Implement $f(x_0, x_1)$ as a template function to allow
computing both the value of the function and the gradient.

```cpp
template<typename T> T f(const std::vector<T>& x) {
    return log(x.at(0)) + x.at(0) * x.at(1) - sin(x.at(1));
}
```

Using the [`nabla::Tensor`](nablagrad/tensor.hpp) structure and the [`nabla::grad()`](nablagrad/core.hpp)
function the gradient $\nabla f(x_0, x_1)$ can be easily computed and evaluated at some vector $x$ as

```cpp
std::vector<double> x = {2.0, -3.0};

auto grad_f = nabla::grad(f<nabla::Tensor>);
std::cout << "∇f(x) = " << grad_f(x) << std::endl;
```

Running these example as `./examples/reverse_mode_gradient` we obtain

```
∇f(x) = [5.5, 1.7163378145367738092]
```

Although less efficiently (see [Baydin et al. (2018)](https://arxiv.org/abs/1502.05767)), gradient computation
can also be performed in forward-mode by using [`nabla::Dual`](nablagrad/dual.hpp) and
[`nabla::grad_forward()`](nablagrad/core.hpp) instead of `nabla::Tensor` and `nabla::grad()`.

### Forward-mode partial differentiation

Let $\mathbf{x} = [x_0, x_1, \ldots, x_n]^\intercal$ be a vector. The $\ell_2$-norm of $\mathbf{x}$ is defined as
$\lVert\mathbf{x}\rVert_2^2\triangleq\langle\mathbf{x}, \mathbf{x}\rangle$. Partial
differentiation (for instance, $\partial /\partial x_0 \lVert\mathbf{x}\rVert_2^2$)
can be performed by declaring all elements in $\mathbf{x}$ (which are the variables of the $\ell_2$-norm function)
as `nabla::Dual` and setting the adjoint to $1.0$ of the variable to be differentiated with respect to.

The $\ell_2$-norm can be implemented as
```cpp
template<typename T> T l2norm(const std::vector<T>& x) {
    T t{0};
    for (T xi : x) t += xi * xi;
    return sqrt(t);
}
```

Then partial differentiation is performed as follows.

```cpp
nabla::Dual x0{-4.3, 1.0};
nabla::Dual x1{6.2};
nabla::Dual x2{5.8};
std::vector<nabla::Dual> x = {x0, x1, x2};

nabla::Dual dl2norm_x0 = l2norm<nabla::Dual>(x);
std::cout << "∂l2norm/∂x0 = " << dl2norm_x0.get_adjoint() << std::endl;
```

Running this example as `./examples/forward_mode_partial_diff` we obtain

```
∂l2norm/∂x0 = -0.451831
```

The `nabla::grad_forward()` function can be used in order to compute the gradient of a function in
forward-mode.

---

## Installation and usage

*nablagrad* can be installed by running

```bash
git clone https://github.com/rixsilverith/nablagrad.git && cd nablagrad
make install # may need to be run with root priviledges (sudo)
```

> **Note** As of now, only Linux is properly supported. For other operating system manual installation
is required.

Once installed, *nablagrad* must be linked to your project as a static library using your
compiler of choice (for instance, with the `-lnablagrad` flag in `g++`). Then the library can
be included as `#include <nablagrad/nabla.h>`.

### Compiling from source

*nablagrad* can be compiled from source by running `make build`. A `libnablagrad.a` static library
file will be generated inside the `build` directory.

### Requirements

A compiler supporting, at least, C++17 is needed in order to compile *nablagrad* from source.

---

## License

*nablagrad* is licensed under the MIT License. See [LICENSE](LICENSE) for more information. A copy of the
license can be found along with the code.

---

## References

- Baydin, A. G.; Pearlmutter, B. A.; Radul, A. A.; Siskind, J. M. (2018). *Automatic differentiation in machine learning: a survey*. [https://arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767)
- Wengert, R. E. (1964). *A simple automatic derivative evaluation program*. Comm. ACM. 7 (8): 463–464. [[doi]](https://doi.org/10.1145%2F355586.364791)
- *Automatic differentiation*. [Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation).
