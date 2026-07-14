#include "ctranslate2/ops/gumbel_max.h"
#include "ctranslate2/ops/multinomial.h"
#include "ctranslate2/ops/topk.h"
#include "ctranslate2/ops/topp_mask.h"

#include <algorithm>
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
      StorageView selection_mask(x.shape(), DataType::INT8, D);
      int8_t* is_selected = selection_mask.data<int8_t>();
      const dim_t depth = x.dim(-1);
      const dim_t rows = x.size() / depth;
      const dim_t k = _k;
      const size_t wg = sycl_backend::work_group_size();
      auto& queue = sycl_backend::get_queue();

      queue.fill(is_selected, static_cast<int8_t>(0), x.size());

      queue.submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> local_values(wg, handler);
        ::sycl::local_accessor<IndexT, 1> local_indices(wg, handler);
        handler.parallel_for(::sycl::nd_range<1>(rows * wg, wg), [=](::sycl::nd_item<1> item) {
          const dim_t row = item.get_group(0);
          const size_t lid = item.get_local_id(0);
          for (dim_t rank = 0; rank < k; ++rank) {
            float best = -std::numeric_limits<float>::infinity();
            IndexT best_index = static_cast<IndexT>(depth);
            for (dim_t col = lid; col < depth; col += wg) {
              if (is_selected[row * depth + col])
                continue;

              const float value = sycl_backend::to_float(input[row * depth + col]);
              if (is_better_candidate(value,
                                      static_cast<IndexT>(col),
                                      best,
                                      best_index,
                                      static_cast<IndexT>(depth))) {
                best = value;
                best_index = static_cast<IndexT>(col);
              }
            }
            local_values[lid] = best;
            local_indices[lid] = best_index;
            item.barrier(::sycl::access::fence_space::local_space);
            for (size_t stride = wg / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                const float other = local_values[lid + stride];
                const IndexT other_index = local_indices[lid + stride];
                if (is_better_candidate(other,
                                        other_index,
                                        local_values[lid],
                                        local_indices[lid],
                                        static_cast<IndexT>(depth))) {
                  local_values[lid] = other;
                  local_indices[lid] = other_index;
                }
              }
              item.barrier(::sycl::access::fence_space::local_space);
            }
            if (lid == 0) {
              const IndexT selected = local_indices[0];
              result_values[row * k + rank] = sycl_backend::from_float<DT>(local_values[0]);
              result_indices[row * k + rank] = selected;
              is_selected[row * depth + selected] = 1;
            }
            item.barrier(::sycl::access::fence_space::global_and_local);
          }
        });
      });

      // Keep the selection mask alive until the final rank is written.
      queue.wait_and_throw();
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
