//
// Created by Keanu Lim on 7/1/26.
//

#include <gtest/gtest.h>

// forward declare so we don't need to include the whole header
void geMM(const float* A, const float* B, float* C, int M, int N, int K);

TEST(GemmTest, TwoByTwo) {
    // A = [[1, 2],   B = [[5, 6],
    //      [3, 4]]        [7, 8]]
    //
    // C = [[1*5 + 2*7, 1*6 + 2*8],   = [[19, 22],
    //      [3*5 + 4*7, 3*6 + 4*8]]      [43, 50]]

    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0, 0, 0, 0};

    geMM(A, B, C, 2, 2, 2);

    EXPECT_NEAR(C[0], 19.0f, 1e-5f);
    EXPECT_NEAR(C[1], 22.0f, 1e-5f);
    EXPECT_NEAR(C[2], 43.0f, 1e-5f);
    EXPECT_NEAR(C[3], 50.0f, 1e-5f);
}

TEST(GemmTest, NonSquare) {
    // A = [[1, 2, 3],   (2x3)
    //      [4, 5, 6]]
    //
    // B = [[7,  8],     (3x2)
    //      [9,  10],
    //      [11, 12]]
    //
    // C = [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12],   = [[58,  64],
    //      [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]      [139, 154]]

    float A[6] = {1, 2, 3, 4, 5, 6};
    float B[6] = {7, 8, 9, 10, 11, 12};
    float C[4] = {0, 0, 0, 0};

    geMM(A, B, C, 2, 2, 3);  // M=2, N=2, K=3

    EXPECT_NEAR(C[0],  58.0f, 1e-5f);
    EXPECT_NEAR(C[1],  64.0f, 1e-5f);
    EXPECT_NEAR(C[2], 139.0f, 1e-5f);
    EXPECT_NEAR(C[3], 154.0f, 1e-5f);
}
