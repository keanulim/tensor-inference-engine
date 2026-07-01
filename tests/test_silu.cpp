//
// Created by Keanu Lim on 7/1/26.
//
#include <gtest/gtest.h>
#include <cmath>

void siLU(const float* in, float* out, int n);

TEST(SiluTest, BasicSilu) {
    // silu(x) = x * sigmoid(x) = x / (1 + e^-x)
    // silu(0)  = 0 * 0.5 = 0.0
    // silu(1)  = 1 * sigmoid(1) = 1 / (1 + e^-1) = 0.7311
    // silu(-1) = -1 * sigmoid(-1) = -1 / (1 + e^1) = -0.2689

    float in[3]  = {0.0f, 1.0f, -1.0f};
    float out[3] = {0, 0, 0};

    siLU(in, out, 3);

    EXPECT_NEAR(out[0],  0.0000f, 1e-4f);
    EXPECT_NEAR(out[1],  0.7311f, 1e-4f);
    EXPECT_NEAR(out[2], -0.2689f, 1e-4f);
}