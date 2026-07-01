//
// Created by Keanu Lim on 6/24/26.
//

#ifndef TENSOR_ENGINE_KERNELS_H
#define TENSOR_ENGINE_KERNELS_H

#include <cstddef>

void geMM(const float* A, const float* B, float* C, int M, int N, int K);
void reLU(const float* in, float* out, int n);
void siLU(const float* in, float* out, int n);
void softmax(const float* in, float* out, int n);
void layerNorm(const float* in, float* out, int d, const float* weight, const float* bias, float eps = 1e-5f);
void embedding_lookup(const size_t* token_ids, size_t seq_len, const float* weight, size_t d_model, float* out);

#endif //TENSOR_ENGINE_KERNELS_H
