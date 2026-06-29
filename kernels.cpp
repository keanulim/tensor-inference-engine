//
// Created by Keanu Lim on
//
#include <cmath>

void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                if (k == 0) C[i * N + j] = 0.0f;
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
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

void softmax(const float* in, float* out, int n) {
    float  max_val = in[0];
    for (int i = 1; i < n; i ++) {
        if (in[i] > max_val) {
            max_val = in[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = std::exp(in[i]-max_val);
        sum+=out[i];
    }

    for (int i = 0; i < n; i++) {
        out[i]/=sum;
    }

}