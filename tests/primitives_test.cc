#include <limits>

#include "test_utils.h"
#include "ctranslate2/ops/softmax.h"
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
TEST(SYCLWhisperTimestampTest, SuppressesTextForSelectedWinningRows) {
  if (get_device_count(Device::SYCL) == 0)
    GTEST_SKIP() << "No SYCL device is available";

  StorageView logits(
    {3, 8},
    std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8,
                       11, 12, 13, 14, 15, 16, 17, 18,
                       21, 22, 23, 24, 25, 26, 27, 28},
    Device::SYCL);
  const StorageView log_probs(
    {3, 8},
    std::vector<float>{-4, -3, -2, -5, -6, -0.2f, -1, -2,
                       -0.1f, -2, -3, -4, -5, -4, -5, -6,
                       -4, -3, -2, -5, -6, -0.2f, -1, -2},
    Device::SYCL);

  primitives<Device::SYCL>::suppress_text_if_timestamp(
    logits.data<float>(),
    log_probs.data<float>(),
    /*rows=*/3,
    /*depth=*/8,
    /*num_text_tokens=*/5,
    /*timestamp_begin=*/5,
    /*num_timestamp_tokens=*/3,
    /*check_rows=*/uint64_t(0b011),
    /*disable_value=*/-100.f);

  const StorageView expected(
    {3, 8},
    std::vector<float>{-100, -100, -100, -100, -100, 6, 7, 8,
                       11, 12, 13, 14, 15, 16, 17, 18,
                       21, 22, 23, 24, 25, 26, 27, 28});
  expect_storage_eq(logits, expected);
}

TEST(SYCLWhisperTimestampTest, SupportsFloat16AndHighestMaskBit) {
  if (get_device_count(Device::SYCL) == 0
      || !mayiuse_float16(Device::SYCL))
    GTEST_SKIP() << "No FP16-capable SYCL device is available";

  std::vector<float> scores(64 * 8, -10.f);
  std::vector<float> expected = scores;
  scores[63 * 8 + 5] = -0.1f;
  scores[63 * 8 + 6] = -1.f;
  scores[63 * 8 + 7] = -2.f;
  expected = scores;
  std::fill(expected.begin() + 63 * 8,
            expected.begin() + 63 * 8 + 5,
            -100.f);

  StorageView logits
    = StorageView({64, 8}, scores, Device::SYCL).to(DataType::FLOAT16);
  primitives<Device::SYCL>::suppress_text_if_timestamp(
    logits.data<float16_t>(),
    logits.data<float16_t>(),
    /*rows=*/64,
    /*depth=*/8,
    /*num_text_tokens=*/5,
    /*timestamp_begin=*/5,
    /*num_timestamp_tokens=*/3,
    /*check_rows=*/uint64_t(1) << 63,
    /*disable_value=*/float16_t(-100.f));

  expect_storage_eq(logits.to_float32(), StorageView({64, 8}, expected), 1e-3f);
}

TEST(SYCLWhisperTimestampTest, PreservesFloat16LogSoftMaxBoundaryDecision) {
  if (get_device_count(Device::SYCL) == 0
      || !mayiuse_float16(Device::SYCL))
    GTEST_SKIP() << "No FP16-capable SYCL device is available";

  // The raw timestamp logit sum is below the text maximum. LogSoftMax shifts
  // these values to about -8, where FLOAT16 rounding makes their maxima tie;
  // the second timestamp probability then makes the timestamp sum win.
  constexpr dim_t num_text_tokens = 4000;
  constexpr dim_t num_timestamp_tokens = 2;
  constexpr dim_t depth = num_text_tokens + num_timestamp_tokens;
  std::vector<float> scores(depth, 4.26953125f);
  scores[num_text_tokens] = 4.265625f;
  scores[num_text_tokens + 1] = -8.2578125f;

  StorageView logits
    = StorageView({1, depth}, scores, Device::SYCL).to(DataType::FLOAT16);
  primitives<Device::SYCL>::suppress_text_if_timestamp(
    logits.data<float16_t>(),
    logits.data<float16_t>(),
    /*rows=*/1,
    depth,
    num_text_tokens,
    /*timestamp_begin=*/num_text_tokens,
    num_timestamp_tokens,
    /*check_rows=*/1,
    /*disable_value=*/float16_t(-100.f));

  const StorageView result = logits.to_float32().to(Device::CPU);
  for (dim_t i = 0; i < num_text_tokens; ++i)
    ASSERT_EQ(result.at<float>(i), -100.f);
  EXPECT_EQ(result.at<float>(num_text_tokens), scores[num_text_tokens]);
  EXPECT_EQ(result.at<float>(num_text_tokens + 1), scores[num_text_tokens + 1]);
}

TEST(SYCLWhisperTimestampTest, PreservesFloat32StrictComparisonBoundary) {
  if (get_device_count(Device::SYCL) == 0)
    GTEST_SKIP() << "No SYCL device is available";

  const std::vector<float> scores{0.0011705458f, 0.f, -6.7497468f};

  // Compute the decision made by the unfused path. This vector is on a
  // one-ULP boundary on Arc B580, so it also verifies that the fused device
  // logarithm preserves the strict comparison made by host logsumexp.
  const StorageView legacy_logits({1, 3}, scores, Device::SYCL);
  StorageView legacy_log_probs(DataType::FLOAT32, Device::SYCL);
  ops::LogSoftMax()(legacy_logits, legacy_log_probs);
  const float legacy_text_max
    = primitives<Device::SYCL>::max(legacy_log_probs.data<float>(), 1);
  const float legacy_timestamp_log_prob
    = primitives<Device::SYCL>::logsumexp(legacy_log_probs.data<float>() + 1, 2);
  const bool legacy_suppresses = legacy_timestamp_log_prob > legacy_text_max;

  StorageView logits({1, 3}, scores, Device::SYCL);
  primitives<Device::SYCL>::suppress_text_if_timestamp(
    logits.data<float>(),
    logits.data<float>(),
    /*rows=*/1,
    /*depth=*/3,
    /*num_text_tokens=*/1,
    /*timestamp_begin=*/1,
    /*num_timestamp_tokens=*/2,
    /*check_rows=*/1,
    /*disable_value=*/-100.f);

  std::vector<float> expected = scores;
  if (legacy_suppresses)
    expected[0] = -100.f;
  expect_storage_eq(logits, StorageView({1, 3}, expected));
}

TEST(SYCLWhisperTimestampTest, DoesNotSuppressOnExactProbabilityTie) {
  if (get_device_count(Device::SYCL) == 0)
    GTEST_SKIP() << "No SYCL device is available";

  const std::vector<float> scores{0.f, 0.f};
  StorageView logits({1, 2}, scores, Device::SYCL);
  primitives<Device::SYCL>::suppress_text_if_timestamp(
    logits.data<float>(),
    logits.data<float>(),
    /*rows=*/1,
    /*depth=*/2,
    /*num_text_tokens=*/1,
    /*timestamp_begin=*/1,
    /*num_timestamp_tokens=*/1,
    /*check_rows=*/1,
    /*disable_value=*/-100.f);

  expect_storage_eq(logits, StorageView({1, 2}, scores));
}

TEST(SYCLWhisperTimestampTest, RejectsInvalidRanges) {
  EXPECT_THROW(
    primitives<Device::SYCL>::suppress_text_if_timestamp<float>(
      nullptr, nullptr, 65, 8, 5, 5, 3, 1, -100.f),
    std::invalid_argument);
  EXPECT_THROW(
    primitives<Device::SYCL>::suppress_text_if_timestamp<float>(
      nullptr, nullptr, 1, 8, 5, 5, 4, 1, -100.f),
    std::invalid_argument);
  EXPECT_THROW(
    primitives<Device::SYCL>::suppress_text_if_timestamp<float>(
      nullptr,
      nullptr,
      1,
      8,
      5,
      5,
      std::numeric_limits<dim_t>::max(),
      1,
      -100.f),
    std::invalid_argument);
}

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
