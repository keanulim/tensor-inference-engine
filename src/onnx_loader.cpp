// Created by Keanu Lim
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>
#include "onnx.pb.h"

// Simple flat buffer to hold weights
struct WeightStore {
    std::unordered_map<std::string, std::vector<float>> weights;
    std::unordered_map<std::string, std::vector<int>>   shapes;

    void load(const std::string& name, const std::vector<int>& shape, const float* data, size_t n) {
        shapes[name] = shape;
        weights[name] = std::vector<float>(data, data + n);
    }
};

// Simple tensor: shape + flat data pointer
struct Tensor {
    std::vector<float> data;
    std::vector<int>   shape;
};

// GEMM: C = A @ B.T + bias  (ONNX Gemm with transB=1 by default)
void gemm(const float* A, const float* B, const float* bias,
          float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];  // B is transposed
            }
            C[i * N + j] = sum;
        }
    }
}

// ReLU
void relu(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

int main() {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // Load the ONNX file
    onnx::ModelProto model;
    std::ifstream in("/Users/keanulim/projects/tensor-engine/model_inline.onnx", std::ios::binary);
    if (!in) { std::cerr << "Cannot open toy.onnx\n"; return 1; }
    if (!model.ParseFromIstream(&in)) { std::cerr << "Failed to parse\n"; return 1; }

    const auto& graph = model.graph();

    // Load weights into WeightStore
    WeightStore store;
    for (const auto& init : graph.initializer()) {
        std::vector<int> shape;
        size_t n = 1;
        for (auto d : init.dims()) { shape.push_back((int)d); n *= d; }

        std::vector<float> values(n);
        if (!init.raw_data().empty()) {
            // stored as raw bytes
            const float* data = reinterpret_cast<const float*>(init.raw_data().data());
            values.assign(data, data + n);
        } else if (init.float_data_size() > 0) {
            // stored as repeated float
            values.assign(init.float_data().begin(), init.float_data().end());
        } else {
            std::cerr << "Warning: weight " << init.name() << " has no data!\n";
        }

        store.load(init.name(), shape, values.data(), n);
        std::cout << "Loaded weight: " << init.name() << " shape: [";
        for (int i = 0; i < (int)shape.size(); i++)
            std::cout << shape[i] << (i+1<(int)shape.size()?",":"");
        std::cout << "]\n";
    }

    // Build a name -> tensor map for intermediate activations
    std::unordered_map<std::string, Tensor> tensors;

    // Seed the input: batch=1, input_dim=64, all ones
    Tensor input;
    input.shape = {1, 64};
    input.data.assign(64, 1.0f);
    tensors["input"] = input;

    // Execute each node in order
    for (const auto& node : graph.node()) {
        const std::string& op = node.op_type();
        std::cout << "Running op: " << op << "\n";

        if (op == "Gemm") {
            const std::string& in_name     = node.input(0);
            const std::string& weight_name = node.input(1);
            const std::string& bias_name   = node.input(2);
            const std::string& out_name    = node.output(0);

            auto& X = tensors[in_name];
            auto& W = store.weights[weight_name];
            auto& b = store.weights[bias_name];
            auto& Wshape = store.shapes[weight_name];

            int M = X.shape[0];
            int N = Wshape[0];   // output features
            int K = Wshape[1];   // input features

            Tensor out;
            out.shape = {M, N};
            out.data.resize(M * N);
            gemm(X.data.data(), W.data(), b.data(), out.data.data(), M, N, K);
            tensors[out_name] = out;

        } else if (op == "Relu") {
            const std::string& in_name  = node.input(0);
            const std::string& out_name = node.output(0);

            Tensor out = tensors[in_name];  // copy
            relu(out.data.data(), (int)out.data.size());
            tensors[out_name] = out;

        } else {
            std::cout << "  (skipping unknown op: " << op << ")\n";
        }
    }

    // Print the final output
    const std::string& last_out = graph.node(graph.node_size()-1).output(0);
    auto& result = tensors[last_out];
    std::cout << "\nOutput logits (" << result.shape[0]
              << "x" << result.shape[1] << "):\n";
    for (float v : result.data)
        std::cout << v << " ";
    std::cout << "\n";

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}