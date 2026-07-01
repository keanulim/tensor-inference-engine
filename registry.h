#ifndef TENSOR_ENGINE_REGISTRY_H
#define TENSOR_ENGINE_REGISTRY_H

#include <unordered_map>
#include <string>
#include <functional>
#include <cstddef>

struct OpArgs {
    const float* A   = nullptr;
    const float* B   = nullptr;
    float*       out = nullptr;
    const float* weight = nullptr;
    const float* bias   = nullptr;
    const size_t* token_ids = nullptr;
    int M = 0, N = 0, K = 0;
    int n = 0, d = 0;
    size_t seq_len = 0, d_model = 0;
};

using OpRegistry = std::unordered_map<std::string, std::function<void(const OpArgs&)>>;

OpRegistry build_registry();

#endif //TENSOR_ENGINE_REGISTRY_H
