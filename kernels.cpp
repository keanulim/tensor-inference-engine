//
// Created by Keanu Lim on 6/24/26.export CPLUS_INCLUDE_PATH="$(brew --prefix llvm)/include/c++/v1"
export LDFLAGS="-L$(brew --prefix llvm)/lib"
//
#include <cmath>

void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] + B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void relu(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

void silu(const float* in, float* out, int n) {
    for ( int i = 0; i < n; i++) {
        out[i] = in[i] / (1.0f + std::exp(-in[i]));
    }
}

void softmax(const float* in, float* out, int in) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += in[i];
    }
    for (int i = 0; i < n; i++) {
        out[i] = in[i]/sum;
    }

}