// Created by Keanu Lim
#ifndef TENSOR_ENGINE_TENSOR_H
#define TENSOR_ENGINE_TENSOR_H

#include <vector>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int>   shape;

    Tensor() = default;

    Tensor(std::vector<int> shape_in) : shape(shape_in) {
        int total = 1;
        for (int dim : shape) total *= dim;
        data.resize(total, 0.0f);
    }

    float& at(int i, int j) {
        return data[i * shape[1] + j];
    }

    int numel() const {
        int total = 1;
        for (int d : shape) total *= d;
        return total;
    }
};

#endif