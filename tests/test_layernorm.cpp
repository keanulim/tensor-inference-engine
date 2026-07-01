//
// Created by Keanu Lim on 7/1/26.
//
#include <gtest/gtest.h>

void layerNorm(const float* in, float* out, int d, const float* weight, const float* bias, float eps = 1e-5f);

TEST(LayernormTest, BasicNormalization) {
    float in[4]     = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4]    = {0, 0, 0, 0};
    float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float bias[4]   = {0.0f, 0.0f, 0.0f, 0.0f};

    layerNorm(in, out, 4, weight, bias);

    EXPECT_NEAR(out[0], -1.3416f, 1e-3f);
    EXPECT_NEAR(out[1], -0.4472f, 1e-3f);
    EXPECT_NEAR(out[2],  0.4472f, 1e-3f);
    EXPECT_NEAR(out[3],  1.3416f, 1e-3f);
}

TEST(LayernormTest, WeightAndBiasApplied) {
    float in[4]     = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4]    = {0, 0, 0, 0};
    float weight[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float bias[4]   = {1.0f, 1.0f, 1.0f, 1.0f};

    layerNorm(in, out, 4, weight, bias);

    // normalized * 2 + 1
    EXPECT_NEAR(out[0], -1.6832f, 1e-3f);  // -1.3416 * 2 + 1
    EXPECT_NEAR(out[1],  0.1056f, 1e-3f);  // -0.4472 * 2 + 1
    EXPECT_NEAR(out[2],  1.8944f, 1e-3f);  //  0.4472 * 2 + 1
    EXPECT_NEAR(out[3],  3.6832f, 1e-3f);  //  1.3416 * 2 + 1
}