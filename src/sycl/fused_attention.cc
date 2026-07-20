#include "fused_attention.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "sycl/helpers.h"

namespace ctranslate2 {
  namespace sycl_backend {
    namespace {

      constexpr size_t attention_work_group_size = 256;
      constexpr dim_t attention_head_dim = 64;
      constexpr dim_t original_attention_batch_size = 5;
      constexpr dim_t maximum_attention_batch_size = 64;
      constexpr dim_t attention_num_heads = 20;
      constexpr dim_t maximum_profitable_key_length = 320;
      constexpr size_t value_reduction_width = 4;
      static_assert(attention_work_group_size
                    == attention_head_dim * value_reduction_width);

      size_t required_local_memory(const dim_t key_length) {
        return static_cast<size_t>(key_length) * sizeof(::sycl::half)
               + attention_head_dim * sizeof(::sycl::half)
               + attention_work_group_size * sizeof(float);
      }

      bool device_supports_fused_attention(const int device_index,
                                           const dim_t key_length) {
        struct Cache {
          int device_index = -2;
          bool supported = false;
          size_t local_memory_size = 0;
        };
        thread_local Cache cache;
        if (cache.device_index != device_index) {
          const ::sycl::device& device = get_device(device_index);
          const bool supported
            = device.has(::sycl::aspect::fp16)
              && device.get_info<::sycl::info::device::max_work_group_size>()
                   >= attention_work_group_size
              && device.get_info<
                   ::sycl::info::device::max_work_item_sizes<1>>()[0]
                   >= attention_work_group_size;
          const size_t local_memory_size
            = device.get_info<::sycl::info::device::local_mem_size>();
          cache.device_index = device_index;
          cache.supported = supported;
          cache.local_memory_size = local_memory_size;
        }
        return cache.supported
               && required_local_memory(key_length) <= cache.local_memory_size;
      }

      bool can_run_fused_single_query_attention_fp16(const dim_t batch_size,
                                                     const dim_t num_heads,
                                                     const dim_t key_length,
                                                     const dim_t head_dim) {
        const int device_index = get_device_index();
        const IntelGpuModel model = get_intel_gpu_model(device_index);
        if (batch_size <= 0
            || batch_size > maximum_attention_batch_size
            || num_heads != attention_num_heads
            || key_length <= 0
            || key_length > maximum_profitable_key_length
            || head_dim != attention_head_dim
            || !is_fused_attention_profitable(model,
                                              batch_size,
                                              key_length))
          return false;

        if (!device_supports_fused_attention(device_index, key_length))
          return false;

        // Every work-group owns one independent [batch, head] row. Keep the
        // flattened launch and pointer offsets representable even if this
        // internal entry point is called with dimensions not backed by a
        // StorageView.
        const size_t batch = static_cast<size_t>(batch_size);
        const size_t heads = static_cast<size_t>(num_heads);
        const size_t keys = static_cast<size_t>(key_length);
        const size_t head = static_cast<size_t>(head_dim);
        const size_t maximum = std::numeric_limits<size_t>::max();
        if (batch > maximum / heads || keys > maximum / head)
          return false;
        const size_t rows = batch * heads;
        const size_t keys_per_row = keys * head;
        return rows <= maximum / attention_work_group_size
               && rows <= maximum / head
               && rows <= maximum / keys_per_row;
      }

      class FusedSingleQueryAttentionFP16Kernel;

    }

    bool is_fused_attention_profitable(const IntelGpuModel model,
                                       const dim_t batch_size,
                                       const dim_t key_length) {
      if (batch_size <= 0
          || batch_size > maximum_attention_batch_size
          || key_length <= 0
          || key_length > maximum_profitable_key_length)
        return false;

      if (model == IntelGpuModel::ArcB580) {
        if (batch_size <= 5)
          return key_length <= 320;
        if (batch_size <= 10)
          return key_length <= 128;
        if (batch_size <= 20)
          return key_length <= 64;
        return key_length <= 16;
      }

      if (model == IntelGpuModel::ArcB390) {
        // B390's oneMKL crossover happens much earlier than B580's. The
        // scalar fused kernel remains useful for short decoder caches, with
        // one repeatable 16-token window at 20 flattened beam rows.
        if (batch_size <= 4)
          return key_length <= 64;
        if (batch_size == original_attention_batch_size)
          return key_length <= 96;
        if (batch_size == 20)
          return key_length <= 16;
        return batch_size <= maximum_attention_batch_size
               && key_length <= 8;
      }

      if (model == IntelGpuModel::Arc140V) {
        // Measured Lunar Lake crossovers. Rows are flattened decoder batches
        // including the beam dimension, so typical Whisper batches 1/2/4/8
        // arrive here as 5/10/20/40 rows.
        if (batch_size <= 2)
          return key_length <= 64;
        if (batch_size <= 16)
          return key_length <= 32;
        if (batch_size <= 48)
          return key_length <= 16;
        return key_length <= 8;
      }

      // Preserve the original beam-5 dispatch on untuned SYCL products.
      return batch_size == original_attention_batch_size
             && key_length <= maximum_profitable_key_length;
    }

    bool supports_fused_single_query_attention_fp16(
      const FusedSingleQueryAttentionFP16Config& config) {
      if (config.batch_size <= 0
          || config.batch_size > maximum_attention_batch_size
          || config.num_heads != attention_num_heads
          || config.query_length != 1
          || config.key_length <= 0
          || config.key_length > maximum_profitable_key_length
          || config.head_dim != attention_head_dim
          || config.has_mask
          || config.returns_attention
          || config.has_relative_position
          || config.has_alibi)
        return false;
      return can_run_fused_single_query_attention_fp16(config.batch_size,
                                                       config.num_heads,
                                                       config.key_length,
                                                       config.head_dim);
    }

    void fused_single_query_attention_fp16(const float16_t* queries,
                                           const float16_t* keys,
                                           const float16_t* values,
                                           float16_t* output,
                                           const dim_t batch_size,
                                           const dim_t num_heads,
                                           const dim_t key_length,
                                           const dim_t head_dim,
                                           const float scale) {
      if (!queries || !keys || !values || !output)
        throw std::invalid_argument("Fused attention received a null pointer");
      if (!std::isfinite(scale))
        throw std::invalid_argument("Fused attention scale must be finite");
      if (!can_run_fused_single_query_attention_fp16(
            batch_size, num_heads, key_length, head_dim))
        throw std::invalid_argument("Unsupported fused single-query attention shape");

      const auto* device_queries = device_cast(queries);
      const auto* device_keys = device_cast(keys);
      const auto* device_values = device_cast(values);
      auto* device_output = device_cast(output);
      const size_t rows = static_cast<size_t>(batch_size * num_heads);
      const size_t keys_per_row = static_cast<size_t>(key_length * head_dim);

      get_queue().submit([&](::sycl::handler& handler) {
        // The same local allocation first holds rounded logits and is then
        // overwritten by rounded softmax probabilities.
        ::sycl::local_accessor<::sycl::half, 1> weights(
          static_cast<size_t>(key_length), handler);
        ::sycl::local_accessor<::sycl::half, 1> local_query(
          static_cast<size_t>(head_dim), handler);
        ::sycl::local_accessor<float, 1> scratch(attention_work_group_size,
                                                 handler);

        handler.parallel_for<FusedSingleQueryAttentionFP16Kernel>(
          ::sycl::nd_range<1>(rows * attention_work_group_size,
                              attention_work_group_size),
          [=](::sycl::nd_item<1> item) {
            const size_t row = item.get_group(0);
            const size_t lid = item.get_local_id(0);
            const size_t query_offset = row * attention_head_dim;
            const size_t kv_offset = row * keys_per_row;

            if (lid < attention_head_dim)
              local_query[lid] = device_queries[query_offset + lid];
            item.barrier(::sycl::access::fence_space::local_space);

            float thread_maximum = -std::numeric_limits<float>::infinity();
            for (size_t key = lid; key < static_cast<size_t>(key_length);
                 key += attention_work_group_size) {
              float dot = 0.f;
#pragma unroll
              for (size_t d = 0; d < attention_head_dim; ++d) {
                dot = ::sycl::fma(static_cast<float>(local_query[d]),
                                  static_cast<float>(
                                    device_keys[kv_offset + key * attention_head_dim + d]),
                                  dot);
              }
              const ::sycl::half score(dot * scale);
              weights[key] = score;
              thread_maximum = ::sycl::fmax(thread_maximum,
                                             static_cast<float>(score));
            }

            scratch[lid] = thread_maximum;
            item.barrier(::sycl::access::fence_space::local_space);
            for (size_t stride = attention_work_group_size / 2; stride > 0;
                 stride >>= 1) {
              if (lid < stride)
                scratch[lid] = ::sycl::fmax(scratch[lid],
                                             scratch[lid + stride]);
              item.barrier(::sycl::access::fence_space::local_space);
            }
            const float maximum = scratch[0];
            item.barrier(::sycl::access::fence_space::local_space);

            float thread_sum = 0.f;
            for (size_t key = lid; key < static_cast<size_t>(key_length);
                 key += attention_work_group_size) {
              thread_sum += ::sycl::exp(static_cast<float>(weights[key])
                                        - maximum);
            }
            scratch[lid] = thread_sum;
            item.barrier(::sycl::access::fence_space::local_space);
            for (size_t stride = attention_work_group_size / 2; stride > 0;
                 stride >>= 1) {
              if (lid < stride)
                scratch[lid] += scratch[lid + stride];
              item.barrier(::sycl::access::fence_space::local_space);
            }
            const float denominator = scratch[0];
            item.barrier(::sycl::access::fence_space::local_space);

            for (size_t key = lid; key < static_cast<size_t>(key_length);
                 key += attention_work_group_size) {
              const float probability
                = ::sycl::exp(static_cast<float>(weights[key]) - maximum)
                  / denominator;
              weights[key] = ::sycl::half(probability);
            }
            item.barrier(::sycl::access::fence_space::local_space);

            // Four lanes cooperate on each of the 64 output channels.  This
            // shortens the longest V accumulation dependency chain while all
            // arithmetic remains in FP32 until the final FP16 store.
            // Keep each subgroup's V loads contiguous: reduction part is the
            // outer dimension and output channel is the inner dimension.
            const size_t d = lid % attention_head_dim;
            const size_t part = lid / attention_head_dim;
            float value_sum = 0.f;
            for (size_t key = part; key < static_cast<size_t>(key_length);
                 key += value_reduction_width) {
              value_sum = ::sycl::fma(
                static_cast<float>(weights[key]),
                static_cast<float>(
                  device_values[kv_offset + key * attention_head_dim + d]),
                value_sum);
            }
            scratch[lid] = value_sum;
            item.barrier(::sycl::access::fence_space::local_space);

            if (part == 0) {
              const float sum01 = scratch[d]
                                  + scratch[attention_head_dim + d];
              const float sum23 = scratch[2 * attention_head_dim + d]
                                  + scratch[3 * attention_head_dim + d];
              device_output[query_offset + d] = ::sycl::half(sum01 + sum23);
            }
          });
      });
    }

  }
}
