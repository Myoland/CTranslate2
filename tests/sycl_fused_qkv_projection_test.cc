#include "test_utils.h"

#include <random>
#include <vector>

#include "ctranslate2/ops/bias_add.h"
#include "ctranslate2/ops/gemm.h"
#include "sycl/device_info.h"
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

TEST(IntelGpuModelTest, ClassifiesTunedArcProductsByDeviceId) {
  EXPECT_EQ(sycl_backend::intel_gpu_model_from_device_id(0xe20b),
            sycl_backend::IntelGpuModel::ArcB580);
  for (const uint32_t device_id : {0xb080, 0xb082, 0xb084, 0xb086}) {
    EXPECT_EQ(sycl_backend::intel_gpu_model_from_device_id(device_id),
              sycl_backend::IntelGpuModel::ArcB390);
  }

  EXPECT_EQ(sycl_backend::intel_gpu_model_from_device_id(0xb081),
            sycl_backend::IntelGpuModel::Other);
  EXPECT_EQ(sycl_backend::intel_gpu_model_from_device_id(0x64a0, 64),
            sycl_backend::IntelGpuModel::Arc140V);
  EXPECT_EQ(sycl_backend::intel_gpu_model_from_device_id(0x64a0, 56),
            sycl_backend::IntelGpuModel::Other);
  EXPECT_EQ(sycl_backend::intel_gpu_model_from_device_id(0x64a0),
            sycl_backend::IntelGpuModel::Other);
  for (const uint32_t device_id : {0x6420, 0x64b0, 0x64a1}) {
    EXPECT_EQ(sycl_backend::intel_gpu_model_from_device_id(device_id, 64),
              sycl_backend::IntelGpuModel::Other);
  }
  EXPECT_EQ(sycl_backend::intel_gpu_model_from_device_id(0x7d55),
            sycl_backend::IntelGpuModel::Other);
}

TEST(FusedQKVProjectionPolicyTest, UsesProductSpecificRowBoundaries) {
  using sycl_backend::IntelGpuModel;
  using sycl_backend::is_fused_qkv_projection_profitable;

  EXPECT_TRUE(is_fused_qkv_projection_profitable(IntelGpuModel::ArcB580, 1));
  EXPECT_TRUE(is_fused_qkv_projection_profitable(IntelGpuModel::ArcB580, 80));
  EXPECT_FALSE(is_fused_qkv_projection_profitable(IntelGpuModel::ArcB580, 81));

  EXPECT_FALSE(is_fused_qkv_projection_profitable(IntelGpuModel::ArcB390, 4));
  EXPECT_TRUE(is_fused_qkv_projection_profitable(IntelGpuModel::ArcB390, 5));
  EXPECT_TRUE(is_fused_qkv_projection_profitable(IntelGpuModel::ArcB390, 20));
  EXPECT_FALSE(is_fused_qkv_projection_profitable(IntelGpuModel::ArcB390, 21));

  EXPECT_TRUE(is_fused_qkv_projection_profitable(IntelGpuModel::Arc140V, 1));
  EXPECT_TRUE(is_fused_qkv_projection_profitable(IntelGpuModel::Arc140V, 20));
  EXPECT_FALSE(is_fused_qkv_projection_profitable(IntelGpuModel::Arc140V, 21));
  EXPECT_FALSE(is_fused_qkv_projection_profitable(IntelGpuModel::Other, 5));
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

  const sycl_backend::IntelGpuModel model
    = sycl_backend::get_intel_gpu_model();
  for (const dim_t supported_rows : {1, 5, 10, 16, 20, 35, 40, 64, 80}) {
    config.rows = supported_rows;
    EXPECT_EQ(sycl_backend::supports_fused_qkv_projection_fp16(config),
              sycl_backend::is_fused_qkv_projection_profitable(
                model, supported_rows));
  }
}

TEST(SYCLFusedQKVProjectionTest, MatchesOneMKLBiasAndSplit) {
  if (!can_test_fused_qkv())
    GTEST_SKIP() << "The specialized Arc FP16 matrix shape is unavailable";

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
  const sycl_backend::IntelGpuModel model
    = sycl_backend::get_intel_gpu_model();
  const float tolerance
    = (model == sycl_backend::IntelGpuModel::ArcB390
       || model == sycl_backend::IntelGpuModel::Arc140V)
        ? 2e-4f
        : 5e-4f;

  for (const dim_t test_rows : {1, 5, 10, 16, 17, 20, 35, 40, 64, 80}) {
    SCOPED_TRACE(test_rows);
    if (!sycl_backend::supports_fused_qkv_projection_fp16(
          {test_rows, input_size, output_size}))
      continue;
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

    expect_storage_eq(output1.to_float32(), expected1.to_float32(), tolerance);
    expect_storage_eq(output2.to_float32(), expected2.to_float32(), tolerance);
    expect_storage_eq(output3.to_float32(), expected3.to_float32(), tolerance);
  }
}
