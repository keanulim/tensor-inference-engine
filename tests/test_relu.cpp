//
// Created by Keanu Lim on 7/1/26.
//

#include <gtest/gtest.h>

void reLU(const float* in, float* out, int n);

TEST(ReluTest, BasicRelu) {
    float in[4]  = {-2.0f, -0.5f, 0.0f, 3.0f};
    float out[4] = {0, 0, 0, 0};

    reLU(in, out, 4);

    EXPECT_NEAR(out[0], 0.0f, 1e-5f);
    EXPECT_NEAR(out[1], 0.0f, 1e-5f);
    EXPECT_NEAR(out[2], 0.0f, 1e-5f);
    EXPECT_NEAR(out[3], 3.0f, 1e-5f);
}

TEST(ReluTest, AllPositive) {
    float in[3]  = {1.0f, 2.0f, 3.0f};
    float out[3] = {0, 0, 0};

    reLU(in, out, 3);

    EXPECT_NEAR(out[0], 1.0f, 1e-5f);
    EXPECT_NEAR(out[1], 2.0f, 1e-5f);
    EXPECT_NEAR(out[2], 3.0f, 1e-5f);
}