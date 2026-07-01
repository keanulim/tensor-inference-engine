//
// Created by Keanu Lim on 7/1/26.
//
#include <gtest/gtest.h>

void embedding_lookup(const size_t* token_ids, size_t seq_len, const float* weight, size_t d_model, float* out);

TEST(EmbeddingTest, SingleToken) {
    // vocab=3, d_model=4
    // weight matrix (row-major):
    //   token 0 → [1, 2, 3, 4]
    //   token 1 → [5, 6, 7, 8]
    //   token 2 → [9, 10, 11, 12]

    float weight[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    size_t token_ids[1] = {1};  // look up token 1
    float out[4] = {0, 0, 0, 0};

    embedding_lookup(token_ids, 1, weight, 4, out);

    EXPECT_NEAR(out[0], 5.0f, 1e-5f);
    EXPECT_NEAR(out[1], 6.0f, 1e-5f);
    EXPECT_NEAR(out[2], 7.0f, 1e-5f);
    EXPECT_NEAR(out[3], 8.0f, 1e-5f);
}

TEST(EmbeddingTest, MultipleTokens) {
    // look up tokens [2, 0] → should get rows 2 and 0 concatenated in out

    float weight[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    size_t token_ids[2] = {2, 0};
    float out[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    embedding_lookup(token_ids, 2, weight, 4, out);

    // token 2
    EXPECT_NEAR(out[0],  9.0f, 1e-5f);
    EXPECT_NEAR(out[1], 10.0f, 1e-5f);
    EXPECT_NEAR(out[2], 11.0f, 1e-5f);
    EXPECT_NEAR(out[3], 12.0f, 1e-5f);
    // token 0
    EXPECT_NEAR(out[4],  1.0f, 1e-5f);
    EXPECT_NEAR(out[5],  2.0f, 1e-5f);
    EXPECT_NEAR(out[6],  3.0f, 1e-5f);
    EXPECT_NEAR(out[7],  4.0f, 1e-5f);
}