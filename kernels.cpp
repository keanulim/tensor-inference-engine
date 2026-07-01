//
// Created by Keanu Lim on
//
#include <cmath>
#include <algorithm>


void geMM(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}



void reLU(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

void siLU(const float* in, float* out, int n) {
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

void layerNorm(const float* in, float* out, int d, const float* weight, const float* bias,
    float eps = 1e-5f) {

    float sum = 0.0f;

    for (int i = 0; i < d; i++) {
        sum += in[i];
    }

    float mean = sum / d;

    float sum_var = 0.0f;
    for (int i = 0; i < d; i++) {
        sum_var += (in[i] - mean) * (in[i] - mean);
    }

    float var = sum_var/d;
    float inverted_std= 1.0f /std::sqrt(var + eps);

    for (int i = 0;  i < d; i++) {
        out[i] = (in[i] - mean) * inverted_std * weight[i] + bias[i];
    }
}

void embedding_lookup(const size_t* token_ids, size_t seq_len, const float* weight, size_t d_model,
    float* out) {

    for (int i = 0; i < seq_len; i++) {
        const float* src = weight + token_ids[i] * d_model; //row we want

        float* dst = out + i * d_model; //where it goes

        for (int j = 0; j < d_model; j++) {
            dst[j] = src[j];
        }
    }



}
