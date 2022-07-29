[![License](https://img.shields.io/github/license/rixsilverith/nablagrad)](https://mit-license.org/)
![Cpp Version](https://img.shields.io/badge/C%2B%2B-17-blue)

# nablagrad

Automatic differentiation is a technique to automatically compute and evaluate the partial derivatives (and therefore
gradient) of a given mathematical function.

*nablagrad* is yet another automatic differentiation library written in C++ as a proof of concept. Despite its main purpose being
learning how autodiff works and how it may be implemented, it should be a fast and reliable tool to perform first-order
differentiation.

> **Note** As of now, *nablagrad* only supports forward-mode autodiff, but it is expected to also provide
support for reverse-mode autodiff at some point.

---

## Installation

For now, only local installation is available. Clone this repo and move the `nablagrad` folder into your
project directory. Import the [`nablagrad/lib/nablagrad.hpp`](lib/nablagrad.hpp) header file.

---

## Basic usage example

In this example we compute the gradient of the $\ell_2$-norm evaluated at some real-valued vector $x$.
After importing the `nablagrad.hpp` header file accordingly, we define the `l2norm` function as

```cpp
nabla::Dual l2norm(const nabla::DualVec& x) {
    nabla::Dual n{0.};
    for (auto i : x) n += i * i;
    return sqrt(n);
}
```
Note that the `<cmath>` header must also be included in order to use the `sqrt` function.

Now, we compute and evaluate the gradient as follows.

```cpp
nabla::RealVec x = { .5, 1.5 }; // nabla::RealVec is just an alias for std::vector<double>

auto grad_norm = nabla::grad(l2norm);
std::cout << grad_norm(x) << std::endl;
```
```
[0.316228, 0.948683]
```

---

## License

*nablagrad* is licensed under the MIT License. See [LICENSE](LICENSE) for more information. A copy of the license can be found along with the code.

---

## References

- Baydin, A. G.; Pearlmutter, B. A.; Radul, A. A.; Siskind, J. M. (2018). *Automatic differentiation in machine learning: a survey*. [https://arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767)
