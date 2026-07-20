#include "test_utils.h"

#include <cmath>
#include <utility>
#include <vector>

#include "ctranslate2/ops/matmul.h"
#include "ctranslate2/ops/softmax.h"
#include "sycl/fused_attention.h"

namespace {

  constexpr dim_t default_batch_size = 5;
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

  bool supports_fused_attention(const dim_t batch_size,
                                const dim_t key_length) {
    if (get_device_count(Device::SYCL) == 0
        || !mayiuse_float16(Device::SYCL))
      return false;
    const sycl_backend::FusedSingleQueryAttentionFP16Config config{
      batch_size, num_heads, 1, key_length, head_dim};
    return sycl_backend::supports_fused_single_query_attention_fp16(config);
  }

  void test_fused_attention(const dim_t batch_size, const dim_t key_length) {
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

TEST(FusedAttentionPolicyTest, UsesProductSpecificKeyBoundaries) {
  using sycl_backend::IntelGpuModel;
  using sycl_backend::is_fused_attention_profitable;

  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::ArcB580, 5, 320));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::ArcB580, 10, 128));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::ArcB580, 20, 64));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::ArcB580, 64, 16));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::ArcB580, 64, 17));

  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 1, 64));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 1, 65));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 5, 96));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 5, 97));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 10, 8));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 10, 9));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 20, 16));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 20, 17));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 64, 8));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::ArcB390, 64, 9));

  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::Arc140V, 2, 64));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::Arc140V, 2, 65));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::Arc140V, 16, 32));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::Arc140V, 16, 33));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::Arc140V, 48, 16));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::Arc140V, 48, 17));
  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::Arc140V, 64, 8));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::Arc140V, 64, 9));

  EXPECT_TRUE(is_fused_attention_profitable(IntelGpuModel::Other, 5, 320));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::Other, 5, 321));
  EXPECT_FALSE(is_fused_attention_profitable(IntelGpuModel::Other, 10, 8));
}

TEST(SYCLFusedAttentionTest, MatchesUnfusedPathAtK32) {
  if (!supports_fused_attention(default_batch_size, 32))
    GTEST_SKIP() << "The default fused-attention shape is unavailable";
  test_fused_attention(default_batch_size, 32);
  for (const dim_t batch_size : {1, 10, 20}) {
    if (supports_fused_attention(batch_size, 32))
      test_fused_attention(batch_size, 32);
  }
}

TEST(SYCLFusedAttentionTest, MatchesProductSpecificBoundaries) {
  if (get_device_count(Device::SYCL) == 0
      || !mayiuse_float16(Device::SYCL))
    GTEST_SKIP() << "No FP16-capable SYCL device is available";

  bool tested_shape = false;
  for (const auto [batch_size, key_length]
       : std::vector<std::pair<dim_t, dim_t>>{
           {1, 64}, {5, 96}, {10, 8}, {10, 32}, {16, 32},
           {20, 16}, {48, 16}, {64, 8}}) {
    if (supports_fused_attention(batch_size, key_length)) {
      test_fused_attention(batch_size, key_length);
      tested_shape = true;
    }
  }
  if (!tested_shape)
    GTEST_SKIP() << "No product-specific fused-attention boundary is available";
}

TEST(SYCLFusedAttentionTest, MatchesUnfusedPathAtK64) {
  if (!supports_fused_attention(11, 64))
    GTEST_SKIP() << "The expanded batched kernel is unavailable";
  for (const dim_t batch_size : {11, 20})
    test_fused_attention(batch_size, 64);
}

TEST(SYCLFusedAttentionTest, MatchesUnfusedPathAtK16) {
  bool tested_shape = false;
  for (const dim_t batch_size : {40, 64}) {
    if (supports_fused_attention(batch_size, 16)) {
      test_fused_attention(batch_size, 16);
      tested_shape = true;
    }
  }
  if (!tested_shape)
    GTEST_SKIP() << "The expanded batched kernel is unavailable";
}

TEST(SYCLFusedAttentionTest, MatchesUnfusedPathAtK128) {
  if (!supports_fused_attention(default_batch_size, 128))
    GTEST_SKIP() << "The fused-attention shape is outside this product's boundary";
  test_fused_attention(default_batch_size, 128);
}

TEST(SYCLFusedAttentionTest, MatchesBatchedUnfusedPathAtK128) {
  if (!supports_fused_attention(10, 128))
    GTEST_SKIP() << "The expanded batched kernel is unavailable";
  test_fused_attention(10, 128);
}

TEST(SYCLFusedAttentionTest, MatchesUnfusedPathAtK320) {
  if (!supports_fused_attention(default_batch_size, 320))
    GTEST_SKIP() << "The fused-attention shape is outside this product's boundary";
  test_fused_attention(default_batch_size, 320);
}

TEST(SYCLFusedAttentionTest, DispatchPredicateIsStrict) {
  if (get_device_count(Device::SYCL) == 0
      || !mayiuse_float16(Device::SYCL))
    GTEST_SKIP() << "No FP16-capable SYCL device is available";

  sycl_backend::FusedSingleQueryAttentionFP16Config config{
    default_batch_size, num_heads, 1, 16, head_dim};
  if (!sycl_backend::supports_fused_single_query_attention_fp16(config))
    GTEST_SKIP() << "The base fused-attention shape is unavailable";

  const sycl_backend::IntelGpuModel model
    = sycl_backend::get_intel_gpu_model();
  for (const auto [batch_size, key_length]
       : std::vector<std::pair<dim_t, dim_t>>{
           {1, 8}, {1, 64}, {5, 96}, {5, 320}, {6, 8}, {10, 9},
           {20, 16}, {20, 17}, {40, 8}, {40, 16}, {64, 8}, {64, 16}}) {
    config.batch_size = batch_size;
    config.key_length = key_length;
    EXPECT_EQ(sycl_backend::supports_fused_single_query_attention_fp16(config),
              sycl_backend::is_fused_attention_profitable(
                model, batch_size, key_length));
  }

  config.batch_size = 65;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));

  config.batch_size = default_batch_size;
  config.key_length = 320;
  EXPECT_EQ(sycl_backend::supports_fused_single_query_attention_fp16(config),
            sycl_backend::is_fused_attention_profitable(
              model, default_batch_size, 320));

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
  config.batch_size = 0;
  EXPECT_FALSE(sycl_backend::supports_fused_single_query_attention_fp16(config));
  config.batch_size = default_batch_size;
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
