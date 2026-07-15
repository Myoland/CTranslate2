#include "test_utils.h"

#include <cmath>
#include <vector>

#include "ctranslate2/ops/matmul.h"
#include "ctranslate2/ops/softmax.h"
#include "sycl/fused_attention.h"

namespace {

  constexpr dim_t batch_size = 5;
  constexpr dim_t num_heads = 20;
  constexpr dim_t head_dim = 64;
  constexpr float query_scale = 0.125f;

  std::vector<float> make_values(const dim_t size, const float phase) {
    std::vector<float> values(size);
    for (dim_t i = 0; i < size; ++i) {
      const float x = static_cast<float>(i);
      values[i] = 0.3f * std::sin(x * 0.013f + phase)
                  + 0.05f * std::cos(x * 0.031f - phase);
    }
    return values;
  }

  void test_fused_attention(const dim_t key_length) {
    if (get_device_count(Device::SYCL) == 0
        || !mayiuse_float16(Device::SYCL))
      GTEST_SKIP() << "No FP16-capable SYCL device is available";

    const Shape query_shape{batch_size, num_heads, 1, head_dim};
    const Shape key_value_shape{batch_size, num_heads, key_length, head_dim};
    const StorageView queries
      = StorageView(query_shape, make_values(compute_size(query_shape), 0.1f),
                    Device::SYCL).to(DataType::FLOAT16);
    const StorageView keys
      = StorageView(key_value_shape,
                    make_values(compute_size(key_value_shape), 0.7f),
                    Device::SYCL).to(DataType::FLOAT16);
    const StorageView values
      = StorageView(key_value_shape,
                    make_values(compute_size(key_value_shape), -0.4f),
                    Device::SYCL).to(DataType::FLOAT16);

    StorageView attention(DataType::FLOAT16, Device::SYCL);
    ops::MatMul(/*trans_a=*/false, /*trans_b=*/true, query_scale)(
      queries, keys, attention);
    ops::SoftMax()(attention, attention);
    StorageView expected(DataType::FLOAT16, Device::SYCL);
    ops::MatMul()(attention, values, expected);

    StorageView output(query_shape, DataType::FLOAT16, Device::SYCL);
    sycl_backend::fused_single_query_attention_fp16(
      queries.data<float16_t>(),
      keys.data<float16_t>(),
      values.data<float16_t>(),
      output.data<float16_t>(),
      batch_size,
      num_heads,
      key_length,
      head_dim,
      query_scale);

    // The fused V reduction uses four partial FP32 sums.  Both paths round
    // logits and probabilities to FP16; the changed accumulation order stays
    // within this absolute tolerance for representative Whisper inputs.
    expect_storage_eq(output.to_float32(), expected.to_float32(), 1e-4f);
  }

}

TEST(SYCLFusedAttentionTest, MatchesUnfusedPathAtK32) {
  test_fused_attention(32);
}

TEST(SYCLFusedAttentionTest, MatchesUnfusedPathAtK128) {
  test_fused_attention(128);
}

TEST(SYCLFusedAttentionTest, MatchesUnfusedPathAtK320) {
  test_fused_attention(320);
}

TEST(SYCLFusedAttentionTest, DispatchPredicateIsStrict) {
  if (get_device_count(Device::SYCL) == 0
      || !mayiuse_float16(Device::SYCL))
    GTEST_SKIP() << "No FP16-capable SYCL device is available";

  sycl_backend::FusedSingleQueryAttentionFP16Config config{
    batch_size, num_heads, 1, 32, head_dim};
  EXPECT_TRUE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.key_length = 320;
  EXPECT_TRUE(sycl_backend::supports_fused_single_query_attention_fp16(config));

  config.key_length = 321;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.key_length = 0;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.key_length = 32;

  config.query_length = 2;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.query_length = 1;
  config.head_dim = 32;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.head_dim = head_dim;
  config.batch_size = 4;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.batch_size = batch_size;
  config.num_heads = 16;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.num_heads = num_heads;

  config.has_mask = true;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.has_mask = false;
  config.returns_attention = true;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.returns_attention = false;
  config.has_relative_position = true;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.has_relative_position = false;
  config.has_alibi = true;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
}
