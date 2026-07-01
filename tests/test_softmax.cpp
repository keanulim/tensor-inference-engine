#include <gtest/gtest.h>
#include <cmath>

void softmax(const float* in, float* out, int n);

TEST(SoftmaxTest, BasicThreeElements) {
    float in[3]  = {1.0f, 2.0f, 3.0f};
    float out[3] = {0, 0, 0};

    softmax(in, out, 3);

    EXPECT_NEAR(out[0], 0.0900f, 1e-4f);
    EXPECT_NEAR(out[1], 0.2447f, 1e-4f);
    EXPECT_NEAR(out[2], 0.6652f, 1e-4f);
}

TEST(SoftmaxTest, AllSameValues) {
    float in[3]  = {1.0f, 1.0f, 1.0f};
    float out[3] = {0, 0, 0};

    softmax(in, out, 3);

    EXPECT_NEAR(out[0], 1.0f/3.0f, 1e-5f);
    EXPECT_NEAR(out[1], 1.0f/3.0f, 1e-5f);
    EXPECT_NEAR(out[2], 1.0f/3.0f, 1e-5f);
}