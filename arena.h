// Created by Keanu Lim
#ifndef TENSOR_ENGINE_ARENA_H
#define TENSOR_ENGINE_ARENA_H

#include <cstdlib>
#include <cstddef>
#include <stdexcept>

class Arena {
    char*  buffer;
    size_t capacity;
    size_t weight_end = 0;  // end of permanent weight region
    size_t bump       = 0;  // current scratch position

public:
    Arena(size_t bytes) {
        buffer   = (char*) malloc(bytes);
        capacity = bytes;
    }

    ~Arena() { free(buffer); }

    // permanent — for weights loaded once at startup
    float* alloc_static(size_t num_floats) {
        size_t bytes = num_floats * sizeof(float);
        if (weight_end + bytes > capacity) throw std::bad_alloc();
        float* ptr = (float*)(buffer + weight_end);
        weight_end += bytes;
        bump = weight_end;
        return ptr;
    }

    // temporary — for activations during forward pass
    float* alloc(size_t num_floats) {
        size_t bytes = num_floats * sizeof(float);
        if (bump + bytes > capacity) throw std::bad_alloc();
        float* ptr = (float*)(buffer + bump);
        bump += bytes;
        return ptr;
    }

    // wipes scratch but keeps weights
    void reset() { bump = weight_end; }
};

#endif