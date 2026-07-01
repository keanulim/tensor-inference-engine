#include "registry.h"
#include "kernels.h"

OpRegistry build_registry() {
    OpRegistry registry;

    registry["gemm"] = [](const OpArgs& a) { geMM(a.A, a.B, a.out, a.M, a.N, a.K); };
    registry["relu"] = [](const OpArgs& a) { reLU(a.A, a.out, a.n); };
    registry["silu"] = [](const OpArgs& a) { siLU(a.A, a.out, a.n); };
    registry["softmax"] = [](const OpArgs& a) { softmax(a.A, a.out, a.n); };
    registry["layernorm"] = [](const OpArgs& a) { layerNorm(a.A, a.out, a.d, a.weight, a.bias); };
    registry["embedding"] = [](const OpArgs& a) { embedding_lookup(a.token_ids, a.seq_len, a.weight, a.d_model, a.out); };

    return registry;
}

