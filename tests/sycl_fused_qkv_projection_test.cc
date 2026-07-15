#include "test_utils.h"

#include <random>
#include <vector>

#include "ctranslate2/ops/bias_add.h"
#include "ctranslate2/ops/gemm.h"
#include "sycl/fused_qkv_projection.h"

namespace {

  constexpr dim_t rows = 5;
  constexpr dim_t max_rows = 80;
  constexpr dim_t input_size = 1280;
  constexpr dim_t output_size = 3840;

  bool can_test_fused_qkv() {
    if (get_device_count(Device::SYCL) == 0
        || !mayiuse_float16(Device::SYCL))
      return false;
    return sycl_backend::supports_fused_qkv_projection_fp16(
      {rows, input_size, output_size});
  }

}

TEST(SYCLFusedQKVProjectionTest, DispatchPredicateIsStrict) {
  if (get_device_count(Device::SYCL) == 0
      || !mayiuse_float16(Device::SYCL))
    GTEST_SKIP() << "No FP16-capable SYCL device is available";

  sycl_backend::FusedQKVProjectionFP16Config config{
    rows, input_size, output_size};
  const bool exact_shape_supported
    = sycl_backend::supports_fused_qkv_projection_fp16(config);

  config.rows = 0;
  EXPECT_FALSE(sycl_backend::supports_fused_qkv_projection_fp16(config));
  config.rows = max_rows + 1;
  EXPECT_FALSE(sycl_backend::supports_fused_qkv_projection_fp16(config));
  config.rows = rows;
  ++config.input_size;
  EXPECT_FALSE(sycl_backend::supports_fused_qkv_projection_fp16(config));
  --config.input_size;
  ++config.output_size;
  EXPECT_FALSE(sycl_backend::supports_fused_qkv_projection_fp16(config));
  --config.output_size;

  if (!exact_shape_supported)
    GTEST_SKIP() << "The exact-shape kernel is intentionally disabled on this device";

  for (const dim_t supported_rows : {1, 5, 10, 16, 20, 35, 40, 64, 80}) {
    config.rows = supported_rows;
    EXPECT_TRUE(sycl_backend::supports_fused_qkv_projection_fp16(config));
  }
}

TEST(SYCLFusedQKVProjectionTest, MatchesOneMKLBiasAndSplit) {
  if (!can_test_fused_qkv())
    GTEST_SKIP() << "The specialized B580 FP16 matrix shape is unavailable";

  std::mt19937 generator(123);
  std::uniform_real_distribution<float> value_distribution(-0.05f, 0.05f);
  std::uniform_real_distribution<float> bias_distribution(-0.01f, 0.01f);
  std::vector<float> weight_values(output_size * input_size);
  std::vector<float> bias_values(output_size);
  for (float& value : weight_values)
    value = value_distribution(generator);
  for (float& value : bias_values)
    value = bias_distribution(generator);

  const StorageView weight
    = StorageView({output_size, input_size}, weight_values)
        .to(DataType::FLOAT16).to(Device::SYCL);
  const StorageView bias
    = StorageView({output_size}, bias_values)
        .to(DataType::FLOAT16).to(Device::SYCL);

  for (const dim_t test_rows : {1, 5, 10, 16, 17, 20, 35, 40, 64, 80}) {
    SCOPED_TRACE(test_rows);
    std::vector<float> input_values(test_rows * input_size);
    for (float& value : input_values)
      value = value_distribution(generator);
    const StorageView input
      = StorageView({test_rows, 1, input_size}, input_values)
          .to(DataType::FLOAT16).to(Device::SYCL);

    StorageView projection(DataType::FLOAT16, Device::SYCL);
    ops::Gemm(/*alpha=*/1, /*beta=*/0, /*trans_a=*/false, /*trans_b=*/true)(
      input, weight, projection);
    StorageView expected1(DataType::FLOAT16, Device::SYCL);
    StorageView expected2(DataType::FLOAT16, Device::SYCL);
    StorageView expected3(DataType::FLOAT16, Device::SYCL);
    ops::BiasAdd().split3(
      projection, bias, expected1, expected2, expected3);

    StorageView output1(DataType::FLOAT16, Device::SYCL);
    StorageView output2(DataType::FLOAT16, Device::SYCL);
    StorageView output3(DataType::FLOAT16, Device::SYCL);
    ASSERT_TRUE(sycl_backend::fused_qkv_projection_fp16(
      input, weight, bias, output1, output2, output3));

    expect_storage_eq(output1.to_float32(), expected1.to_float32(), 5e-4f);
    expect_storage_eq(output2.to_float32(), expected2.to_float32(), 5e-4f);
    expect_storage_eq(output3.to_float32(), expected3.to_float32(), 5e-4f);
  }
}
