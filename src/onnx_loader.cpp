// Created by Keanu Lim
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>
#include "onnx.pb.h"
#include "../arena.h"
#include "../registry.h"
#include "../tensor.h"

onnx::ModelProto load_model(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    onnx::ModelProto model;
    if (!model.ParseFromIstream(&in)) throw std::runtime_error("Failed to parse ONNX file");
    return model;
}

void load_weights(const onnx::ModelProto& model,
                  Arena& arena,
                  std::unordered_map<std::string, Tensor>& tensors) {
    const auto& graph = model.graph();

    for (const auto& init : graph.initializer()) {
        std::vector<int> shape;
        size_t n = 1;
        for (auto d : init.dims()) { shape.push_back((int)d); n *= d; }

        float* dst = arena.alloc_static(n);

        if (!init.raw_data().empty()) {
            const float* src = reinterpret_cast<const float*>(init.raw_data().data());
            std::memcpy(dst, src, n * sizeof(float));
        } else {
            for (size_t i = 0; i < n; i++)
                dst[i] = init.float_data((int)i);
        }

        Tensor t;
        t.shape = shape;
        t.data  = std::vector<float>(dst, dst + n);
        tensors[init.name()] = t;

        std::cout << "Loaded: " << init.name() << " [";
        for (int i = 0; i < (int)shape.size(); i++)
            std::cout << shape[i] << (i+1<(int)shape.size()?",":"");
        std::cout << "]\n";
    }
}

void run_forward(const onnx::ModelProto& model,
                 std::unordered_map<std::string, Tensor>& tensors,
                 Arena& arena) {
    const auto& graph = model.graph();
    OpRegistry registry = build_registry();

    for (const auto& node : graph.node()) {
        const std::string& op = node.op_type();
        std::cout << "Running: " << op << "\n";

        if (op == "Gemm") {
            auto& X      = tensors[node.input(0)];
            auto& W      = tensors[node.input(1)];
            auto& b      = tensors[node.input(2)];
            auto& Wshape = W.shape;

            int M = X.shape[0];
            int N = Wshape[0];
            int K = Wshape[1];

            float* out_ptr = arena.alloc(M * N);

            // GEMM with transB: C = X @ W.T + bias
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float sum = b.data[j];
                    for (int k = 0; k < K; k++)
                        sum += X.data[i*K+k] * W.data[j*K+k];
                    out_ptr[i*N+j] = sum;
                }

            Tensor out;
            out.shape = {M, N};
            out.data  = std::vector<float>(out_ptr, out_ptr + M*N);
            tensors[node.output(0)] = out;

        } else if (op == "Relu") {
            auto& in = tensors[node.input(0)];
            int n = (int)in.data.size();
            float* out_ptr = arena.alloc(n);

            OpArgs args;
            args.A   = in.data.data();
            args.out = out_ptr;
            args.n   = n;
            registry["relu"](args);

            Tensor out;
            out.shape = in.shape;
            out.data  = std::vector<float>(out_ptr, out_ptr + n);
            tensors[node.output(0)] = out;

        } else {
            std::cout << "  skipping: " << op << "\n";
        }
    }
}

int main() {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    try {
        // 100MB arena — enough for this toy model
        Arena arena(100 * 1024 * 1024);

        auto model = load_model("/Users/keanulim/projects/tensor-engine/model_inline.onnx");

        std::unordered_map<std::string, Tensor> tensors;

        load_weights(model, arena, tensors);

        // Seed input: batch=1, dim=64, all ones
        Tensor input;
        input.shape = {1, 64};
        input.data.assign(64, 1.0f);
        tensors["input"] = input;

        run_forward(model, tensors, arena);

        // Print output
        const std::string& last = model.graph().node(
            model.graph().node_size()-1).output(0);
        auto& result = tensors[last];
        std::cout << "\nOutput logits:\n";
        for (float v : result.data) std::cout << v << " ";
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}