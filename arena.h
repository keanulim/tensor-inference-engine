//
// Created by Keanu Lim on 6/18/26.
//

#ifndef UNTITLED6_ARENA_H
#define UNTITLED6_ARENA_H

#include <cstdlib>
#include <cstddef>



class arena {

    char* buffer; //pointer to end of weights
    size_t capacity; //total bytes
    size_t offset; //current bytes allocated
public:
    Arena(size_t bytes) {
        buffer = (char*) malloc(bytes); //one malloc at the beginning to allocate memor
        capacity = bytes;
        offset = 0;
    }

    ~Arena() {
        free(buffer); //runs when arena dies?
    }

    float* alloc(size_t num_floats) { //move pointer x amt of bytes based off num of floats
        size_t bytes = num_floats * sizeof(float);
        float* ptr = (float*)(buffer + offset);
        offset += bytes;
        return ptr;

    }

    void reset() {
        offset = 0;
    }
};


#endif //UNTITLED6_ARENA_H
