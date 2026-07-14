#include "test_utils.h"
#include "ctranslate2/primitives.h"
#include "dispatch.h"

class PrimitiveTest : public ::testing::TestWithParam<Device> {
};

TEST_P(PrimitiveTest, StridedFill) {
  const Device device = GetParam();
  StorageView x({3, 2}, float(0), device);
  StorageView expected({3, 2}, std::vector<float>{1, 0, 1, 0, 1, 0}, device);
  DEVICE_DISPATCH(device, primitives<D>::strided_fill(x.data<float>(), 1.f, 2, 3));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, IndexedFill) {
  const Device device = GetParam();
  StorageView x({6}, float(0), device);
  StorageView ids({3}, std::vector<int32_t>{0, 2, 5}, device);
  StorageView expected({6}, std::vector<float>{1, 0, 1, 0, 0, 1}, device);
  DEVICE_DISPATCH(device, primitives<D>::indexed_fill(x.data<float>(), 1.f, ids.data<int32_t>(), 3));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, LogSumExp) {
  const Device device = GetParam();
  StorageView x({8}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2}, device);
  float result = 0;
  DEVICE_DISPATCH(device, result = primitives<D>::logsumexp(x.data<float>(), x.size()));
  EXPECT_NEAR(result, 2.1908040046691895, 1e-6);
}

TEST_P(PrimitiveTest, PenalizePreviousTokens) {
  const Device device = GetParam();
  const float penalty = 1.2f;
  StorageView scores({2, 4}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2});
  StorageView previous_ids({2, 2}, std::vector<int32_t>{2, 2, 1, 2}, device);
  StorageView previous_scores({2, 2}, std::vector<float>{-1.2, -1.2, 0.5, -1.3}, device);
  StorageView expected = scores;
  expected.at<float>({0, 2}) *= penalty;
  expected.at<float>({1, 1}) /= penalty;
  expected.at<float>({1, 2}) *= penalty;
  scores = scores.to(device);
  DEVICE_DISPATCH(device, primitives<D>::penalize_previous_tokens(scores.data<float>(),
                                                                  previous_scores.data<float>(),
                                                                  previous_ids.data<int32_t>(),
                                                                  penalty,
                                                                  scores.dim(0),
                                                                  previous_ids.dim(1),
                                                                  scores.dim(1)));
  expect_storage_eq(scores, expected);
}

#ifdef CT2_WITH_SYCL
TEST(SYCLFloat16GemmTest, PreservesBetaAndBatchStrides) {
  if (get_device_count(Device::SYCL) == 0
      || !mayiuse_float16(Device::SYCL))
    GTEST_SKIP() << "No FP16-capable SYCL device is available";

  const StorageView broadcast_a = StorageView(
    {2, 3},
    std::vector<float>{1, 2, 3,
                       4, 5, 6},
    Device::SYCL).to(DataType::FLOAT16);
  const StorageView strided_a = StorageView(
    {2, 2, 3},
    std::vector<float>{1, 2, 3,
                       4, 5, 6,
                       1, 2, 3,
                       4, 5, 6},
    Device::SYCL).to(DataType::FLOAT16);
  const StorageView b = StorageView(
    {2, 3, 2},
    std::vector<float>{1, 0,
                       0, 1,
                       1, 1,
                       2, 1,
                       1, 0,
                       0, 1},
    Device::SYCL).to(DataType::FLOAT16);
  const StorageView initial_c = StorageView(
    {2, 2, 2},
    std::vector<float>{0.5f, -0.5f,
                       1.f, -1.f,
                       -0.25f, 0.25f,
                       2.f, -2.f},
    Device::SYCL).to(DataType::FLOAT16);
  const StorageView expected(
    {2, 2, 2},
    std::vector<float>{4.25f, 4.75f,
                       10.5f, 10.5f,
                       3.875f, 4.125f,
                       14.f, 9.f});

  for (const bool broadcast : {false, true}) {
    StorageView c = initial_c;
    const auto& a = broadcast ? broadcast_a : strided_a;
    primitives<Device::SYCL>::gemm_batch_strided(
      /*transpose_a=*/false,
      /*transpose_b=*/false,
      /*m=*/2,
      /*n=*/2,
      /*k=*/3,
      /*alpha=*/1.f,
      a.data<float16_t>(),
      /*lda=*/3,
      /*stridea=*/broadcast ? 0 : 6,
      b.data<float16_t>(),
      /*ldb=*/2,
      /*strideb=*/6,
      /*beta=*/0.5f,
      c.data<float16_t>(),
      /*ldc=*/2,
      /*stridec=*/4,
      /*batch_size=*/2);
    expect_storage_eq(c.to_float32(), expected, 1e-3f);
  }
}

TEST(SYCLFloat16GemmTest, KeepsFP32ScalingForNonRepresentableAlpha) {
  if (get_device_count(Device::SYCL) == 0
      || !mayiuse_float16(Device::SYCL))
    GTEST_SKIP() << "No FP16-capable SYCL device is available";

  const StorageView a = StorageView(
    {1, 1}, std::vector<float>{0.5f}, Device::SYCL).to(DataType::FLOAT16);
  const StorageView b = StorageView(
    {1, 1}, std::vector<float>{1.5f}, Device::SYCL).to(DataType::FLOAT16);
  StorageView c({1, 1}, DataType::FLOAT16, Device::SYCL);
  primitives<Device::SYCL>::gemm(
    /*a_is_packed=*/false,
    /*b_is_packed=*/false,
    /*transpose_a=*/false,
    /*transpose_b=*/false,
    /*m=*/1,
    /*n=*/1,
    /*k=*/1,
    /*alpha=*/0.1f,
    a.data<float16_t>(),
    /*lda=*/1,
    b.data<float16_t>(),
    /*ldb=*/1,
    /*beta=*/0.f,
    c.data<float16_t>(),
    /*ldc=*/1);

  // FP32 alpha produces 0.07501220703125 after the final FP16 rounding.
  // Rounding alpha to FP16 first would instead produce 0.074951171875.
  const StorageView expected({1, 1}, std::vector<float>{0.07501220703125f});
  expect_storage_eq(c.to_float32(), expected);
}
#endif

INSTANTIATE_TEST_SUITE_P(CPU, PrimitiveTest, ::testing::Values(Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, PrimitiveTest, ::testing::Values(Device::CUDA));
#endif
#ifdef CT2_WITH_SYCL
INSTANTIATE_TEST_SUITE_P(SYCL, PrimitiveTest, ::testing::Values(Device::SYCL));
#endif
