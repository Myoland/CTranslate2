#include "ctranslate2/ops/gumbel_max.h"
#include "ctranslate2/ops/multinomial.h"
#include "ctranslate2/ops/topk.h"
#include "ctranslate2/ops/topp_mask.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "sycl/ops_utils.h"
#include "sycl/random.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      template <typename IndexT>
      inline bool value_precedes(const float candidate_value,
                                 const IndexT candidate_index,
                                 const float current_value,
                                 const IndexT current_index) {
        const bool candidate_is_nan = ::sycl::isnan(candidate_value);
        const bool current_is_nan = ::sycl::isnan(current_value);
        if (candidate_is_nan != current_is_nan)
          return !candidate_is_nan;
        if (candidate_is_nan)
          return candidate_index < current_index;
        return candidate_value > current_value
               || (candidate_value == current_value && candidate_index < current_index);
      }

      template <typename IndexT>
      inline bool is_better_candidate(const float candidate_value,
                                      const IndexT candidate_index,
                                      const float current_value,
                                      const IndexT current_index,
                                      const IndexT invalid_index) {
        return candidate_index != invalid_index
               && (current_index == invalid_index
                   || value_precedes(candidate_value,
                                     candidate_index,
                                     current_value,
                                     current_index));
      }

      constexpr size_t max_topk_blocks = 128;
      constexpr size_t topk_items_per_work_item = 8;

      template <size_t K, typename IndexT>
      inline void insert_topk_candidate(const float candidate_value,
                                        const IndexT candidate_index,
                                        std::array<float, K>& top_values,
                                        std::array<IndexT, K>& top_indices,
                                        const IndexT invalid_index) {
        float value = candidate_value;
        IndexT index = candidate_index;

        for (size_t slot = 0; slot < K; ++slot) {
          if (is_better_candidate(value,
                                  index,
                                  top_values[slot],
                                  top_indices[slot],
                                  invalid_index)) {
            const float displaced_value = top_values[slot];
            const IndexT displaced_index = top_indices[slot];
            top_values[slot] = value;
            top_indices[slot] = index;
            value = displaced_value;
            index = displaced_index;
          }
        }
      }

      inline size_t topk_num_blocks(const dim_t depth,
                                    const dim_t k,
                                    const size_t work_group_size) {
        const size_t depth_size = static_cast<size_t>(depth);
        const size_t k_size = static_cast<size_t>(k);
        const size_t items_per_group = work_group_size * topk_items_per_work_item;
        const size_t blocks_for_parallelism
          = (depth_size + items_per_group - 1) / items_per_group;
        // Do not create more partial lists than the input can populate. This
        // keeps the temporary allocation bounded when k approaches depth.
        const size_t blocks_for_candidates = std::max<size_t>(1, depth_size / k_size);
        return std::max<size_t>(1,
                                std::min({max_topk_blocks,
                                          blocks_for_parallelism,
                                          blocks_for_candidates}));
      }

      template <size_t K, typename DT, typename IndexT>
      void submit_specialized_topk(::sycl::queue& queue,
                                   const DT* input,
                                   DT* result_values,
                                   IndexT* result_indices,
                                   DT* partial_values,
                                   IndexT* partial_indices,
                                   const dim_t rows,
                                   const dim_t depth,
                                   const size_t blocks,
                                   const size_t work_group_size) {
        const IndexT invalid_index = static_cast<IndexT>(depth);

        queue.submit([&](::sycl::handler& handler) {
          ::sycl::local_accessor<float, 1> local_values(work_group_size, handler);
          ::sycl::local_accessor<IndexT, 1> local_indices(work_group_size, handler);
          ::sycl::local_accessor<uint32_t, 1> local_lanes(work_group_size, handler);
          handler.parallel_for(
            ::sycl::nd_range<1>(static_cast<size_t>(rows) * blocks * work_group_size,
                                work_group_size),
            [=](::sycl::nd_item<1> item) {
              const size_t group = item.get_group(0);
              const dim_t row = static_cast<dim_t>(group / blocks);
              const size_t block = group % blocks;
              const size_t lid = item.get_local_id(0);
              std::array<float, K> thread_values;
              std::array<IndexT, K> thread_indices;

              for (size_t rank = 0; rank < K; ++rank) {
                thread_values[rank] = -std::numeric_limits<float>::infinity();
                thread_indices[rank] = invalid_index;
              }

              const dim_t first_col
                = static_cast<dim_t>(block * work_group_size + lid);
              const dim_t col_stride
                = static_cast<dim_t>(blocks * work_group_size);
              for (dim_t col = first_col; col < depth; col += col_stride) {
                insert_topk_candidate<K>(
                  sycl_backend::to_float(input[row * depth + col]),
                  static_cast<IndexT>(col),
                  thread_values,
                  thread_indices,
                  invalid_index);
              }

              size_t cursor = 0;
              const size_t partial_offset = group * K;
              for (size_t rank = 0; rank < K; ++rank) {
                local_values[lid] = thread_values[cursor];
                local_indices[lid] = thread_indices[cursor];
                local_lanes[lid] = static_cast<uint32_t>(lid);
                item.barrier(::sycl::access::fence_space::local_space);

                for (size_t stride = work_group_size / 2; stride > 0; stride >>= 1) {
                  if (lid < stride
                      && is_better_candidate(local_values[lid + stride],
                                             local_indices[lid + stride],
                                             local_values[lid],
                                             local_indices[lid],
                                             invalid_index)) {
                    local_values[lid] = local_values[lid + stride];
                    local_indices[lid] = local_indices[lid + stride];
                    local_lanes[lid] = local_lanes[lid + stride];
                  }
                  item.barrier(::sycl::access::fence_space::local_space);
                }

                const uint32_t winner_lane = local_lanes[0];
                const float winner_value = local_values[0];
                const IndexT winner_index = local_indices[0];
                if (lid == winner_lane && cursor + 1 < K)
                  ++cursor;
                if (lid == 0) {
                  partial_values[partial_offset + rank]
                    = sycl_backend::from_float<DT>(winner_value);
                  partial_indices[partial_offset + rank] = winner_index;
                }
                item.barrier(::sycl::access::fence_space::local_space);
              }
            });
        });

        queue.submit([&](::sycl::handler& handler) {
          ::sycl::local_accessor<float, 1> local_values(work_group_size, handler);
          ::sycl::local_accessor<IndexT, 1> local_indices(work_group_size, handler);
          ::sycl::local_accessor<uint32_t, 1> local_lanes(work_group_size, handler);
          handler.parallel_for(
            ::sycl::nd_range<1>(static_cast<size_t>(rows) * work_group_size,
                                work_group_size),
            [=](::sycl::nd_item<1> item) {
              const dim_t row = static_cast<dim_t>(item.get_group(0));
              const size_t lid = item.get_local_id(0);
              std::array<float, K> thread_values;
              std::array<IndexT, K> thread_indices;

              for (size_t rank = 0; rank < K; ++rank) {
                thread_values[rank] = -std::numeric_limits<float>::infinity();
                thread_indices[rank] = invalid_index;
              }

              const size_t candidate_count = blocks * K;
              const size_t row_offset = static_cast<size_t>(row) * candidate_count;
              for (size_t candidate = lid;
                   candidate < candidate_count;
                   candidate += work_group_size) {
                const IndexT candidate_index = partial_indices[row_offset + candidate];
                if (candidate_index == invalid_index)
                  continue;
                insert_topk_candidate<K>(
                  sycl_backend::to_float(partial_values[row_offset + candidate]),
                  candidate_index,
                  thread_values,
                  thread_indices,
                  invalid_index);
              }

              size_t cursor = 0;
              for (size_t rank = 0; rank < K; ++rank) {
                local_values[lid] = thread_values[cursor];
                local_indices[lid] = thread_indices[cursor];
                local_lanes[lid] = static_cast<uint32_t>(lid);
                item.barrier(::sycl::access::fence_space::local_space);

                for (size_t stride = work_group_size / 2; stride > 0; stride >>= 1) {
                  if (lid < stride
                      && is_better_candidate(local_values[lid + stride],
                                             local_indices[lid + stride],
                                             local_values[lid],
                                             local_indices[lid],
                                             invalid_index)) {
                    local_values[lid] = local_values[lid + stride];
                    local_indices[lid] = local_indices[lid + stride];
                    local_lanes[lid] = local_lanes[lid + stride];
                  }
                  item.barrier(::sycl::access::fence_space::local_space);
                }

                const uint32_t winner_lane = local_lanes[0];
                const float winner_value = local_values[0];
                const IndexT winner_index = local_indices[0];
                if (lid == winner_lane && cursor + 1 < K)
                  ++cursor;
                if (lid == 0) {
                  result_values[row * K + rank]
                    = sycl_backend::from_float<DT>(winner_value);
                  result_indices[row * K + rank] = winner_index;
                }
                item.barrier(::sycl::access::fence_space::local_space);
              }
            });
        });
      }

      template <typename DT, typename IndexT>
      void submit_general_topk(::sycl::queue& queue,
                               const DT* input,
                               DT* result_values,
                               IndexT* result_indices,
                               DT* partial_values,
                               IndexT* partial_indices,
                               const dim_t rows,
                               const dim_t depth,
                               const dim_t k,
                               const size_t blocks,
                               const size_t work_group_size) {
        const IndexT invalid_index = static_cast<IndexT>(depth);

        queue.submit([&](::sycl::handler& handler) {
          ::sycl::local_accessor<float, 1> local_values(work_group_size, handler);
          ::sycl::local_accessor<IndexT, 1> local_indices(work_group_size, handler);
          handler.parallel_for(
            ::sycl::nd_range<1>(static_cast<size_t>(rows) * blocks * work_group_size,
                                work_group_size),
            [=](::sycl::nd_item<1> item) {
              const size_t group = item.get_group(0);
              const dim_t row = static_cast<dim_t>(group / blocks);
              const size_t block = group % blocks;
              const size_t lid = item.get_local_id(0);
              float previous_value = -std::numeric_limits<float>::infinity();
              IndexT previous_index = invalid_index;

              for (dim_t rank = 0; rank < k; ++rank) {
                float best = -std::numeric_limits<float>::infinity();
                IndexT best_index = invalid_index;
                const dim_t first_col
                  = static_cast<dim_t>(block * work_group_size + lid);
                const dim_t col_stride
                  = static_cast<dim_t>(blocks * work_group_size);
                for (dim_t col = first_col; col < depth; col += col_stride) {
                  const float value = sycl_backend::to_float(input[row * depth + col]);
                  const IndexT index = static_cast<IndexT>(col);
                  if (rank != 0
                      && (previous_index == invalid_index
                          || !value_precedes(previous_value,
                                             previous_index,
                                             value,
                                             index)))
                    continue;
                  if (is_better_candidate(value,
                                          index,
                                          best,
                                          best_index,
                                          invalid_index)) {
                    best = value;
                    best_index = index;
                  }
                }

                local_values[lid] = best;
                local_indices[lid] = best_index;
                item.barrier(::sycl::access::fence_space::local_space);
                for (size_t stride = work_group_size / 2; stride > 0; stride >>= 1) {
                  if (lid < stride
                      && is_better_candidate(local_values[lid + stride],
                                             local_indices[lid + stride],
                                             local_values[lid],
                                             local_indices[lid],
                                             invalid_index)) {
                    local_values[lid] = local_values[lid + stride];
                    local_indices[lid] = local_indices[lid + stride];
                  }
                  item.barrier(::sycl::access::fence_space::local_space);
                }

                previous_value = local_values[0];
                previous_index = local_indices[0];
                if (lid == 0) {
                  const size_t offset = group * static_cast<size_t>(k)
                                        + static_cast<size_t>(rank);
                  partial_values[offset] = sycl_backend::from_float<DT>(previous_value);
                  partial_indices[offset] = previous_index;
                }
                // All work-items must read the selected threshold before lane
                // 0 can overwrite its reduction slot in the next iteration.
                item.barrier(::sycl::access::fence_space::local_space);
              }
            });
        });

        queue.submit([&](::sycl::handler& handler) {
          ::sycl::local_accessor<float, 1> local_values(work_group_size, handler);
          ::sycl::local_accessor<IndexT, 1> local_indices(work_group_size, handler);
          handler.parallel_for(
            ::sycl::nd_range<1>(static_cast<size_t>(rows) * work_group_size,
                                work_group_size),
            [=](::sycl::nd_item<1> item) {
              const dim_t row = static_cast<dim_t>(item.get_group(0));
              const size_t lid = item.get_local_id(0);
              const size_t candidate_count = blocks * static_cast<size_t>(k);
              const size_t row_offset = static_cast<size_t>(row) * candidate_count;
              float previous_value = -std::numeric_limits<float>::infinity();
              IndexT previous_index = invalid_index;

              for (dim_t rank = 0; rank < k; ++rank) {
                float best = -std::numeric_limits<float>::infinity();
                IndexT best_index = invalid_index;
                for (size_t candidate = lid;
                     candidate < candidate_count;
                     candidate += work_group_size) {
                  const IndexT index = partial_indices[row_offset + candidate];
                  if (index == invalid_index)
                    continue;
                  const float value
                    = sycl_backend::to_float(partial_values[row_offset + candidate]);
                  if (rank != 0
                      && (previous_index == invalid_index
                          || !value_precedes(previous_value,
                                             previous_index,
                                             value,
                                             index)))
                    continue;
                  if (is_better_candidate(value,
                                          index,
                                          best,
                                          best_index,
                                          invalid_index)) {
                    best = value;
                    best_index = index;
                  }
                }

                local_values[lid] = best;
                local_indices[lid] = best_index;
                item.barrier(::sycl::access::fence_space::local_space);
                for (size_t stride = work_group_size / 2; stride > 0; stride >>= 1) {
                  if (lid < stride
                      && is_better_candidate(local_values[lid + stride],
                                             local_indices[lid + stride],
                                             local_values[lid],
                                             local_indices[lid],
                                             invalid_index)) {
                    local_values[lid] = local_values[lid + stride];
                    local_indices[lid] = local_indices[lid + stride];
                  }
                  item.barrier(::sycl::access::fence_space::local_space);
                }

                previous_value = local_values[0];
                previous_index = local_indices[0];
                if (lid == 0) {
                  result_values[row * k + rank]
                    = sycl_backend::from_float<DT>(previous_value);
                  result_indices[row * k + rank] = previous_index;
                }
                item.barrier(::sycl::access::fence_space::local_space);
              }
            });
        });
      }

      template <typename T>
      class TopPSortPolicy;

    }

    template <Device D, typename T, typename IndexT>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      if (_k < 0 || _k > x.dim(-1))
        throw std::invalid_argument("TopK k must be between 0 and the last input dimension");
      if (_k == 0)
        return;

      using DT = sycl_backend::device_type_t<T>;
      const DT* input = sycl_backend::device_cast(x.data<T>());
      DT* result_values = sycl_backend::device_cast(values.data<T>());
      IndexT* result_indices = indices.data<IndexT>();
      const dim_t depth = x.dim(-1);
      const dim_t rows = x.size() / depth;
      const dim_t k = _k;
      if (rows == 0)
        return;
      const size_t wg = sycl_backend::work_group_size();
      const size_t blocks = topk_num_blocks(depth, k, wg);
      StorageView partial_values({rows * static_cast<dim_t>(blocks) * k}, x.dtype(), D);
      StorageView partial_indices({rows * static_cast<dim_t>(blocks) * k},
                                  DataType::INT32,
                                  D);
      DT* partial_values_data
        = sycl_backend::device_cast(partial_values.data<T>());
      IndexT* partial_indices_data = partial_indices.data<IndexT>();
      auto& queue = sycl_backend::get_queue();

#define SUBMIT_SPECIALIZED_TOPK(K)                                     \
      submit_specialized_topk<K>(queue,                                \
                                 input,                                \
                                 result_values,                        \
                                 result_indices,                       \
                                 partial_values_data,                  \
                                 partial_indices_data,                 \
                                 rows,                                 \
                                 depth,                                \
                                 blocks,                               \
                                 wg)

      switch (k) {
      case 10: SUBMIT_SPECIALIZED_TOPK(10); break;
      default:
        submit_general_topk(queue,
                            input,
                            result_values,
                            result_indices,
                            partial_values_data,
                            partial_indices_data,
                            rows,
                            depth,
                            k,
                            blocks,
                            wg);
        break;
      }

#undef SUBMIT_SPECIALIZED_TOPK

      // The in-order queue keeps both stages ordered. StorageView destruction
      // retires these buffers through the SYCL allocator's pending list, so
      // they cannot be recycled while either kernel still references them.
    }

    template <Device D, typename T>
    void TopPMask::compute(const StorageView& input,
                           const StorageView& probs,
                           StorageView& output) const {
      using DT = sycl_backend::device_type_t<T>;
      if (input.size() > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("SYCL TopPMask input is too large to index with INT32");

      const DT* scores = sycl_backend::device_cast(input.data<T>());
      const DT* probabilities = sycl_backend::device_cast(probs.data<T>());
      DT* result = sycl_backend::device_cast(output.data<T>());
      const dim_t depth = input.dim(-1);
      const dim_t rows = input.size() / depth;
      const float p = _p;
      const float mask = _mask_value;
      auto& queue = sycl_backend::get_queue();

      if (p <= 0.f) {
        queue.parallel_for(::sycl::range<1>(input.size()), [=](::sycl::id<1> id) {
          result[id[0]] = sycl_backend::from_float<DT>(mask);
        });
        return;
      }

      StorageView sorted_indices(input.shape(), DataType::INT32, D);
      int32_t* indices = sorted_indices.data<int32_t>();
      queue.parallel_for(::sycl::range<1>(input.size()), [=](::sycl::id<1> id) {
        indices[id[0]] = static_cast<int32_t>(id[0]);
      });

      auto policy = oneapi::dpl::execution::make_device_policy<TopPSortPolicy<T>>(queue);
      std::sort(policy,
                indices,
                indices + input.size(),
                [=](const int32_t left, const int32_t right) {
                  const dim_t left_row = left / depth;
                  const dim_t right_row = right / depth;
                  if (left_row != right_row)
                    return left_row < right_row;
                  return value_precedes(sycl_backend::to_float(probabilities[left]),
                                        left,
                                        sycl_backend::to_float(probabilities[right]),
                                        right);
                });

      queue.parallel_for(::sycl::range<1>(rows), [=](::sycl::id<1> id) {
        const dim_t row = id[0];
        float cumulative = 0.f;
        for (dim_t rank = 0; rank < depth; ++rank) {
          const int32_t selected = indices[row * depth + rank];
          result[selected] = cumulative < p
                               ? scores[selected]
                               : sycl_backend::from_float<DT>(mask);
          cumulative += sycl_backend::to_float(probabilities[selected]);
        }
      });

      // oneDPL and the cutoff kernel consume the temporary index buffer.
      queue.wait_and_throw();
    }

    template <>
    dim_t TopPMask::max_num_classes<Device::SYCL>() {
      return std::numeric_limits<dim_t>::max();
    }

    template <Device D, typename T>
    void GumbelMax::add_gumbel_noise(const StorageView& x, StorageView& y) const {
      using DT = sycl_backend::device_type_t<T>;
      StorageView uniform(x.shape(), DataType::FLOAT32, D);
      sycl_backend::fill_uniform(uniform.data<float>(), uniform.size());
      const DT* input = sycl_backend::device_cast(x.data<T>());
      const float* random = uniform.data<float>();
      DT* output = sycl_backend::device_cast(y.data<T>());
      const dim_t size = x.size();
      sycl_backend::get_queue().parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> id) {
        const dim_t i = id[0];
        const float u = ::sycl::fmax(random[i], std::numeric_limits<float>::min());
        // Keep the same exponential perturbation used by the established CPU
        // and CUDA implementations of this operation.
        const float noise = -::sycl::log(u);
        output[i] = sycl_backend::from_float<DT>(sycl_backend::to_float(input[i]) + noise);
      });
      sycl_backend::get_queue().wait_and_throw();
    }

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      using DT = sycl_backend::device_type_t<T>;
      StorageView uniform(output.shape(), DataType::FLOAT32, D);
      sycl_backend::fill_uniform(uniform.data<float>(), uniform.size());
      const DT* probabilities = sycl_backend::device_cast(input.data<T>());
      const float* random = uniform.data<float>();
      int32_t* result = output.data<int32_t>();
      const dim_t class_size = input.dim(-1);
      const dim_t sample_size = _sample_size;
      const dim_t size = output.size();
      sycl_backend::get_queue().parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> id) {
        const dim_t sample = id[0];
        const dim_t row = sample / sample_size;
        float total = 0.f;
        for (dim_t col = 0; col < class_size; ++col)
          total += ::sycl::fmax(sycl_backend::to_float(probabilities[row * class_size + col]), 0.f);
        const float target = random[sample] * total;
        float cumulative = 0.f;
        int32_t selected = static_cast<int32_t>(class_size - 1);
        for (dim_t col = 0; col < class_size; ++col) {
          cumulative += ::sycl::fmax(sycl_backend::to_float(probabilities[row * class_size + col]), 0.f);
          if (cumulative >= target) {
            selected = static_cast<int32_t>(col);
            break;
          }
        }
        result[sample] = selected;
      });
      sycl_backend::get_queue().wait_and_throw();
    }

#define DECLARE_IMPL(T)                                                 \
    template void TopK::compute<Device::SYCL, T, int32_t>(             \
      const StorageView&, StorageView&, StorageView&) const;           \
    template void TopPMask::compute<Device::SYCL, T>(                  \
      const StorageView&, const StorageView&, StorageView&) const;     \
    template void GumbelMax::add_gumbel_noise<Device::SYCL, T>(        \
      const StorageView&, StorageView&) const;                         \
    template void Multinomial::compute<Device::SYCL, T>(               \
      const StorageView&, StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

  }
}
