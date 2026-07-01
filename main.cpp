#include <iostream>
#include "registry.h"

int main() {
    OpRegistry registry = build_registry();

    // test softmax
    float input[]  = {1.0f, 2.0f, 3.0f};
    float output[] = {0.0f, 0.0f, 0.0f};

    OpArgs args;
    args.A   = input;
    args.out = output;
    args.n   = 3;

    registry["softmax"](args);

    // should print ~0.09, 0.24, 0.66
    for (int i = 0; i < 3; i++) {
        std::cout << output[i] << "\n";
    }

    return 0;
}