#include "test_utils.h"
#include "layers/decoder_cache.h"

TEST(DeferredCacheGatherTest, ComposesMultipleUpdates) {
  constexpr const char* state_name = "deferred_indices";
  layers::DecoderState state;

  layers::detail::defer_cache_gather(
    state,
    StorageView({4}, std::vector<int32_t>{2, 0, 2, 1}),
    state_name);
  layers::detail::defer_cache_gather(
    state,
    StorageView({3}, std::vector<int32_t>{3, 0, 2}),
    state_name);

  const StorageView expected({3}, std::vector<int32_t>{1, 2, 2});
  expect_storage_eq(state.at(state_name), expected);
}

#ifdef CT2_WITH_SYCL

#  include <numeric>

#  include "ctranslate2/ops/concat.h"
#  include "ctranslate2/ops/gather.h"
#  include "sycl/cache_gather_append.h"

namespace {

  StorageView make_values(const Shape& shape, const float offset) {
    std::vector<float> values(compute_size(shape));
    std::iota(values.begin(), values.end(), offset);
    return StorageView(shape, values);
  }

  void test_cache_gather_append(const Shape& previous_shape,
                                const Shape& appended_shape,
                                const dim_t time_axis) {
    if (get_device_count(Device::SYCL) == 0)
      GTEST_SKIP() << "No SYCL device is available";

    const StorageView indices(
      {4}, std::vector<int32_t>{2, 0, 2, 1});
    const StorageView previous_keys = make_values(previous_shape, 0.f);
    const StorageView previous_values = make_values(previous_shape, 100.f);
    const StorageView appended_keys = make_values(appended_shape, 1000.f);
    const StorageView appended_values = make_values(appended_shape, 2000.f);

    StorageView gathered_keys;
    StorageView gathered_values;
    ops::Gather()(previous_keys, indices, gathered_keys);
    ops::Gather()(previous_values, indices, gathered_values);

    StorageView expected_keys;
    StorageView expected_values;
    const ops::Concat concat_op(time_axis);
    concat_op({&gathered_keys, &appended_keys}, expected_keys);
    concat_op({&gathered_values, &appended_values}, expected_values);

    const StorageView previous_keys_device = previous_keys.to(Device::SYCL);
    const StorageView previous_values_device = previous_values.to(Device::SYCL);
    const StorageView indices_device = indices.to(Device::SYCL);
    const StorageView appended_keys_device = appended_keys.to(Device::SYCL);
    const StorageView appended_values_device = appended_values.to(Device::SYCL);
    StorageView output_keys(DataType::FLOAT32, Device::SYCL);
    StorageView output_values(DataType::FLOAT32, Device::SYCL);

    sycl_backend::cache_gather_append(previous_keys_device,
                                      previous_values_device,
                                      indices_device,
                                      appended_keys_device,
                                      appended_values_device,
                                      time_axis,
                                      output_keys,
                                      output_values);

    expect_storage_eq(output_keys, expected_keys);
    expect_storage_eq(output_values, expected_values);
  }

}

TEST(SYCLCacheGatherAppendTest, ReordersAndDuplicatesParentsOnAxis2) {
  test_cache_gather_append(/*previous_shape=*/{3, 2, 2, 2},
                           /*appended_shape=*/{4, 2, 1, 2},
                           /*time_axis=*/2);
}

TEST(SYCLCacheGatherAppendTest, SupportsAxis1) {
  test_cache_gather_append(/*previous_shape=*/{3, 2, 4},
                           /*appended_shape=*/{4, 1, 4},
                           /*time_axis=*/1);
}

#endif
