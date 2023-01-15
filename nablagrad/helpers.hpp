#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>
#include <vector>
#include <string>
#include <random>

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); i++) {
        os << v[i];
        if (i != v.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

template<typename T>
std::vector<T> generate_rand_vect(size_t length) {
    std::vector<T> random_vect;
    random_vect.reserve(length);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> U(0, 1);

    for (size_t i = 0; i < length; i++)
        random_vect.push_back(U(gen));
    return random_vect;
}

#endif
