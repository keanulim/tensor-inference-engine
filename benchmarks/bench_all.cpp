// Created by Keanu Lim
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <numeric>

void geMM(const float* A, const float* B, float* C, int M, int N, int K);
void reLU(const float* in, float* out, int n);
void siLU(const float* in, float* out, int n);
void softmax(const float* in, float* out, int n);
void layerNorm(const float* in, float* out, int d, const float* weight, const float* bias, float eps);
void embedding_lookup(const size_t* token_ids, size_t seq_len, const float* weight, size_t d_model, float* out);

static double time_ms(auto fn) {
    // warmup
    fn();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
}

static void benchmark(const char* name, auto fn) {
    // warmup
    fn();

    std::vector<double> times;
    times.reserve(100);
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double sq_sum = 0.0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / times.size());

    std::cout << name << " | mean: " << mean << " ms | stddev: " << stddev << " ms\n";
}

int main() {
    // GEMM 512x512
    {
        int M = 512, N = 512, K = 512;
        std::vector<float> A(M * K, 1.0f), B(K * N, 1.0f), C(M * N, 0.0f);
        benchmark("GEMM 512x512      ", [&]{ geMM(A.data(), B.data(), C.data(), M, N, K); });
    }

    // ReLU 1M
    {
        int n = 1 << 20;
        std::vector<float> in(n, -0.5f), out(n);
        benchmark("ReLU 1M           ", [&]{ reLU(in.data(), out.data(), n); });
    }

    // SiLU 1M
    {
        int n = 1 << 20;
        std::vector<float> in(n, 0.5f), out(n);
        benchmark("SiLU 1M           ", [&]{ siLU(in.data(), out.data(), n); });
    }

    // Softmax 10k
    {
        int n = 10000;
        std::vector<float> in(n, 1.0f), out(n);
        benchmark("Softmax 10k       ", [&]{ softmax(in.data(), out.data(), n); });
    }

    // LayerNorm d=768
    {
        int d = 768;
        std::vector<float> in(d, 1.0f), out(d), weight(d, 1.0f), bias(d, 0.0f);
        benchmark("LayerNorm d=768   ", [&]{ layerNorm(in.data(), out.data(), d, weight.data(), bias.data(), 1e-5f); });
    }

    // Embedding seq=512 d=768
    {
        size_t seq_len = 512, d_model = 768, vocab = 50257;
        std::vector<float> weight(vocab * d_model, 1.0f), out(seq_len * d_model);
        std::vector<size_t> token_ids(seq_len, 100);
        benchmark("Embedding seq=512 ", [&]{ embedding_lookup(token_ids.data(), seq_len, weight.data(), d_model, out.data()); });
    }

    return 0;
}