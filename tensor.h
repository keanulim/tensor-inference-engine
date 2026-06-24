//
// Created by Keanu Lim on 6/18/26.
//

#ifndef UNTITLED6_TENSOR_H
#define UNTITLED6_TENSOR_H
#include <vector>
#include <iostream>


class tensor {
    std::vector<float> data;
    std::vector<int> shape; //number of elements in each dimension {3,5,4}

    tensor(std::vector<float> shape_in) {
        shape = shape_in;
        //create tensor filled with zeros
        for (int dim : shape) {
            total = total * dim;
        }

        data.resize(total, 0.0f);



    }
    //returns the reference to the value in data
    float& at(int i, int j) {
        int cols = shape[1];
        return data[cols * i + j];
    }
};


#endif //UNTITLED6_TENSOR_H
