#include "ctranslate2/ops/dequantize.h"
#include "ctranslate2/ops/layer_norm.h"
#include "ctranslate2/ops/mean.h"
#include "ctranslate2/ops/quantize.h"
#include "ctranslate2/ops/rms_norm.h"
#include "ctranslate2/ops/softmax.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include "sycl/ops_utils.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      constexpr size_t softmax_work_group_size = 256;
      constexpr size_t softmax_subgroup_size = 32;

      bool supports_softmax_subgroup_reduction(const int device_index,
                                                const dim_t depth) {
        struct Cache {
          int device_index = -2;
          bool supported = false;
          size_t local_memory_size = 0;
        };
        thread_local Cache cache;
        if (cache.device_index != device_index) {
          const ::sycl::device& device = sycl_backend::get_device(device_index);
          const std::string name
            = device.get_info<::sycl::info::device::name>();
          const auto subgroup_sizes
            = device.get_info<::sycl::info::device::sub_group_sizes>();
          cache.device_index = device_index;
          cache.supported
            = name.find("B580") != std::string::npos
              && device.get_info<::sycl::info::device::max_work_group_size>()
                >= softmax_work_group_size
              && std::find(subgroup_sizes.begin(),
                           subgroup_sizes.end(),
                           softmax_subgroup_size) != subgroup_sizes.end();
          cache.local_memory_size
            = device.get_info<::sycl::info::device::local_mem_size>();
        }

        if (!cache.supported || depth <= 0)
          return false;
        const size_t float_capacity = cache.local_memory_size / sizeof(float);
        if (float_capacity < softmax_work_group_size)
          return false;
        return static_cast<size_t>(depth) <= float_capacity
                                              - softmax_work_group_size;
      }

    }

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView* beta,
                            const StorageView* gamma,
                            const StorageView& input,
                            const dim_t,
                            const dim_t outer_size,
                            const dim_t axis_size,
                            const dim_t inner_size,
                            StorageView& output) const {
      using DT = sycl_backend::device_type_t<T>;
      const DT* x = sycl_backend::device_cast(input.data<T>());
      const DT* g = gamma ? sycl_backend::device_cast(gamma->data<T>()) : nullptr;
      const DT* b = beta ? sycl_backend::device_cast(beta->data<T>()) : nullptr;
      DT* y = sycl_backend::device_cast(output.data<T>());
      const dim_t rows = outer_size * inner_size;
      const size_t wg = sycl_backend::work_group_size();
      const float epsilon = _epsilon;

      sycl_backend::get_queue().submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> sums(wg, handler);
        ::sycl::local_accessor<float, 1> squares(wg, handler);
        handler.parallel_for(::sycl::nd_range<1>(rows * wg, wg), [=](::sycl::nd_item<1> item) {
          const dim_t row = item.get_group(0);
          const dim_t outer = row / inner_size;
          const dim_t inner = row % inner_size;
          const size_t lid = item.get_local_id(0);
          float sum = 0.f;
          float square_sum = 0.f;
          for (dim_t k = lid; k < axis_size; k += wg) {
            const dim_t index = (outer * axis_size + k) * inner_size + inner;
            const float value = sycl_backend::to_float(x[index]);
            sum += value;
            square_sum += value * value;
          }
          sums[lid] = sum;
          squares[lid] = square_sum;
          item.barrier(::sycl::access::fence_space::local_space);
          for (size_t stride = wg / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              sums[lid] += sums[lid + stride];
              squares[lid] += squares[lid + stride];
            }
            item.barrier(::sycl::access::fence_space::local_space);
          }
          const float mean = sums[0] / float(axis_size);
          const float variance = ::sycl::fmax(squares[0] / float(axis_size) - mean * mean, 0.f);
          const float inv_std = ::sycl::rsqrt(variance + epsilon);
          for (dim_t k = lid; k < axis_size; k += wg) {
            const dim_t index = (outer * axis_size + k) * inner_size + inner;
            const float gamma_value = g ? sycl_backend::to_float(g[k]) : 1.f;
            const float beta_value = b ? sycl_backend::to_float(b[k]) : 0.f;
            y[index] = sycl_backend::from_float<DT>(
              (sycl_backend::to_float(x[index]) - mean) * inv_std * gamma_value + beta_value);
          }
        });
      });
    }

    template <Device D, typename T>
    void RMSNorm::compute(const StorageView& gamma,
                          const StorageView& input,
                          StorageView& output) const {
      using DT = sycl_backend::device_type_t<T>;
      const DT* x = sycl_backend::device_cast(input.data<T>());
      const DT* g = sycl_backend::device_cast(gamma.data<T>());
      DT* y = sycl_backend::device_cast(output.data<T>());
      const dim_t depth = input.dim(-1);
      const dim_t rows = input.size() / depth;
      const size_t wg = sycl_backend::work_group_size();
      const float epsilon = _epsilon;
      const bool use_residual = _use_residual;

      sycl_backend::get_queue().submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> scratch(wg, handler);
        handler.parallel_for(::sycl::nd_range<1>(rows * wg, wg), [=](::sycl::nd_item<1> item) {
          const dim_t row = item.get_group(0);
          const size_t lid = item.get_local_id(0);
          float sum = 0.f;
          for (dim_t i = lid; i < depth; i += wg) {
            const float value = sycl_backend::to_float(x[row * depth + i]);
            sum += value * value;
          }
          scratch[lid] = sum;
          item.barrier(::sycl::access::fence_space::local_space);
          for (size_t stride = wg / 2; stride > 0; stride >>= 1) {
            if (lid < stride)
              scratch[lid] += scratch[lid + stride];
            item.barrier(::sycl::access::fence_space::local_space);
          }
          const float inv_rms = ::sycl::rsqrt(scratch[0] / float(depth) + epsilon);
          for (dim_t i = lid; i < depth; i += wg) {
            const dim_t index = row * depth + i;
            const float weight = sycl_backend::to_float(g[i]) + (use_residual ? 1.f : 0.f);
            y[index] = sycl_backend::from_float<DT>(sycl_backend::to_float(x[index]) * inv_rms * weight);
          }
        });
      });
    }

    template <Device D, typename T>
    void Mean::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t axis_size,
                       const dim_t inner_size,
                       const bool get_sum,
                       StorageView& output) const {
      using DT = sycl_backend::device_type_t<T>;
      const DT* x = sycl_backend::device_cast(input.data<T>());
      DT* y = sycl_backend::device_cast(output.data<T>());
      const dim_t rows = outer_size * inner_size;
      const size_t wg = sycl_backend::work_group_size();
      sycl_backend::get_queue().submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> scratch(wg, handler);
        handler.parallel_for(::sycl::nd_range<1>(rows * wg, wg), [=](::sycl::nd_item<1> item) {
          const dim_t row = item.get_group(0);
          const dim_t outer = row / inner_size;
          const dim_t inner = row % inner_size;
          const size_t lid = item.get_local_id(0);
          float sum = 0.f;
          for (dim_t k = lid; k < axis_size; k += wg)
            sum += sycl_backend::to_float(x[(outer * axis_size + k) * inner_size + inner]);
          scratch[lid] = sum;
          item.barrier(::sycl::access::fence_space::local_space);
          for (size_t stride = wg / 2; stride > 0; stride >>= 1) {
            if (lid < stride)
              scratch[lid] += scratch[lid + stride];
            item.barrier(::sycl::access::fence_space::local_space);
          }
          if (lid == 0)
            y[row] = sycl_backend::from_float<DT>(get_sum ? scratch[0] : scratch[0] / float(axis_size));
        });
      });
    }

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      using DT = sycl_backend::device_type_t<T>;
      const DT* x = sycl_backend::device_cast(input.data<T>());
      const int32_t* length_data = lengths ? lengths->data<int32_t>() : nullptr;
      DT* y = sycl_backend::device_cast(output.data<T>());
      const dim_t depth = input.dim(-1);
      const dim_t rows = input.size() / depth;
      const size_t wg = sycl_backend::work_group_size();
      const bool log_softmax = _log;

      // The B580 encoder submits many wide FP16 SoftMax rows. Preserve the
      // original binary reduction tree, but finish its last 32 lanes with
      // subgroup shuffles. Cache the FP32 exponentials in local memory so the
      // normalization pass neither reloads the input nor evaluates exp again.
      // Devices that cannot run the exact 256/32 geometry or safely fit the
      // cache retain the generic implementation. LogSoftMax is unchanged.
      if constexpr (std::is_same_v<T, float16_t>) {
        if (!log_softmax
            && wg == softmax_work_group_size
            && supports_softmax_subgroup_reduction(input.device_index(), depth)) {
          sycl_backend::get_queue().submit([&](::sycl::handler& handler) {
            ::sycl::local_accessor<float, 1> scratch(
              softmax_work_group_size, handler);
            ::sycl::local_accessor<float, 1> exponentials(
              static_cast<size_t>(depth), handler);
            handler.parallel_for(
              ::sycl::nd_range<1>(rows * softmax_work_group_size,
                                  softmax_work_group_size),
              [=](::sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(softmax_subgroup_size)]] {
              const dim_t row = item.get_group(0);
              const size_t lid = item.get_local_id(0);
              const ::sycl::sub_group subgroup = item.get_sub_group();
              const size_t lane = subgroup.get_local_linear_id();
              dim_t active = length_data ? dim_t(length_data[row]) : depth;
              active = active < 0 ? 0 : (active > depth ? depth : active);

              float maximum = -std::numeric_limits<float>::infinity();
              for (dim_t i = lid; i < active; i += softmax_work_group_size) {
                maximum = ::sycl::fmax(
                  maximum, sycl_backend::to_float(x[row * depth + i]));
              }
              scratch[lid] = maximum;
              item.barrier(::sycl::access::fence_space::local_space);
              for (size_t stride = softmax_work_group_size / 2;
                   stride >= softmax_subgroup_size;
                   stride >>= 1) {
                if (lid < stride) {
                  scratch[lid] = ::sycl::fmax(scratch[lid],
                                               scratch[lid + stride]);
                }
                item.barrier(::sycl::access::fence_space::local_space);
              }
              maximum = scratch[lid];
              for (size_t stride = softmax_subgroup_size / 2;
                   stride > 0;
                   stride >>= 1) {
                const float other = ::sycl::shift_group_left(
                  subgroup, maximum, static_cast<uint32_t>(stride));
                if (lane < stride)
                  maximum = ::sycl::fmax(maximum, other);
              }
              if (lid == 0)
                scratch[0] = maximum;
              item.barrier(::sycl::access::fence_space::local_space);
              maximum = scratch[0];
              // All subgroups must read the maximum before lane 0 reuses
              // scratch[0] for its partial exponential sum.
              item.barrier(::sycl::access::fence_space::local_space);

              float sum = 0.f;
              for (dim_t i = lid; i < active; i += softmax_work_group_size) {
                const float exponential = ::sycl::exp(
                  sycl_backend::to_float(x[row * depth + i]) - maximum);
                exponentials[i] = exponential;
                sum += exponential;
              }
              scratch[lid] = sum;
              item.barrier(::sycl::access::fence_space::local_space);
              for (size_t stride = softmax_work_group_size / 2;
                   stride >= softmax_subgroup_size;
                   stride >>= 1) {
                if (lid < stride)
                  scratch[lid] += scratch[lid + stride];
                item.barrier(::sycl::access::fence_space::local_space);
              }
              sum = scratch[lid];
              for (size_t stride = softmax_subgroup_size / 2;
                   stride > 0;
                   stride >>= 1) {
                const float other = ::sycl::shift_group_left(
                  subgroup, sum, static_cast<uint32_t>(stride));
                if (lane < stride)
                  sum += other;
              }
              if (lid == 0)
                scratch[0] = sum;
              item.barrier(::sycl::access::fence_space::local_space);
              const float denominator = scratch[0];

              for (dim_t i = lid; i < depth; i += softmax_work_group_size) {
                float value = 0.f;
                if (i < active)
                  value = exponentials[i] / denominator;
                y[row * depth + i] = sycl_backend::from_float<DT>(value);
              }
            });
          });
          return;
        }
      }

      sycl_backend::get_queue().submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> scratch(wg, handler);
        handler.parallel_for(::sycl::nd_range<1>(rows * wg, wg), [=](::sycl::nd_item<1> item) {
          const dim_t row = item.get_group(0);
          const size_t lid = item.get_local_id(0);
          dim_t active = length_data ? dim_t(length_data[row]) : depth;
          active = active < 0 ? 0 : (active > depth ? depth : active);
          float maximum = -std::numeric_limits<float>::infinity();
          for (dim_t i = lid; i < active; i += wg)
            maximum = ::sycl::fmax(maximum, sycl_backend::to_float(x[row * depth + i]));
          scratch[lid] = maximum;
          item.barrier(::sycl::access::fence_space::local_space);
          for (size_t stride = wg / 2; stride > 0; stride >>= 1) {
            if (lid < stride)
              scratch[lid] = ::sycl::fmax(scratch[lid], scratch[lid + stride]);
            item.barrier(::sycl::access::fence_space::local_space);
          }
          maximum = scratch[0];
          item.barrier(::sycl::access::fence_space::local_space);
          float sum = 0.f;
          for (dim_t i = lid; i < active; i += wg)
            sum += ::sycl::exp(sycl_backend::to_float(x[row * depth + i]) - maximum);
          scratch[lid] = sum;
          item.barrier(::sycl::access::fence_space::local_space);
          for (size_t stride = wg / 2; stride > 0; stride >>= 1) {
            if (lid < stride)
              scratch[lid] += scratch[lid + stride];
            item.barrier(::sycl::access::fence_space::local_space);
          }
          const float denominator = scratch[0];
          for (dim_t i = lid; i < depth; i += wg) {
            float value = 0.f;
            if (i < active) {
              const float centered = sycl_backend::to_float(x[row * depth + i]) - maximum;
              value = log_softmax ? centered - ::sycl::log(denominator)
                                  : ::sycl::exp(centered) / denominator;
            }
            y[row * depth + i] = sycl_backend::from_float<DT>(value);
          }
        });
      });
    }

    template <Device D, typename InT, typename OutT>
    void Quantize::quantize(const StorageView& input,
                            StorageView& output,
                            StorageView& scale) const {
      using InDT = sycl_backend::device_type_t<InT>;
      const InDT* x = sycl_backend::device_cast(input.data<InT>());
      OutT* y = output.data<OutT>();
      uint8_t* y_unsigned = reinterpret_cast<uint8_t*>(y);
      float* scales = scale.data<float>();
      const dim_t depth = input.dim(-1);
      const dim_t rows = scale.size();
      const size_t wg = sycl_backend::work_group_size();
      const bool round_before_cast = _round_before_cast;
      const bool shift_to_uint8 = _shift_to_uint8;

      sycl_backend::get_queue().submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> scratch(wg, handler);
        handler.parallel_for(::sycl::nd_range<1>(rows * wg, wg), [=](::sycl::nd_item<1> item) {
          const dim_t row = item.get_group(0);
          const size_t lid = item.get_local_id(0);
          float maximum = 0.f;
          for (dim_t i = lid; i < depth; i += wg)
            maximum = ::sycl::fmax(maximum, ::sycl::fabs(sycl_backend::to_float(x[row * depth + i])));
          scratch[lid] = maximum;
          item.barrier(::sycl::access::fence_space::local_space);
          for (size_t stride = wg / 2; stride > 0; stride >>= 1) {
            if (lid < stride)
              scratch[lid] = ::sycl::fmax(scratch[lid], scratch[lid + stride]);
            item.barrier(::sycl::access::fence_space::local_space);
          }
          const float row_scale = scratch[0] == 0.f ? 1.f : 127.f / scratch[0];
          if (lid == 0)
            scales[row] = row_scale;
          for (dim_t i = lid; i < depth; i += wg) {
            float value = sycl_backend::to_float(x[row * depth + i]) * row_scale;
            if (shift_to_uint8)
              value += 128.f;
            if (round_before_cast)
              value = ::sycl::rint(value);
            if (shift_to_uint8) {
              value = ::sycl::fmax(0.f, ::sycl::fmin(255.f, value));
              y_unsigned[row * depth + i] = static_cast<uint8_t>(value);
            } else {
              value = ::sycl::fmax(-127.f, ::sycl::fmin(127.f, value));
              y[row * depth + i] = static_cast<OutT>(value);
            }
          }
        });
      });
    }

    template <Device D, typename InT, typename OutT>
    void Dequantize::dequantize(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      using OutDT = sycl_backend::device_type_t<OutT>;
      const InT* x = input.data<InT>();
      const float* scales = scale.data<float>();
      OutDT* y = sycl_backend::device_cast(output.data<OutT>());
      const dim_t depth = input.dim(-1);
      const dim_t size = input.size();
      sycl_backend::get_queue().parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> id) {
        const dim_t i = id[0];
        y[i] = sycl_backend::from_float<OutDT>(float(x[i]) / scales[i / depth]);
      });
    }

    template <Device D, typename T>
    void Dequantize::dequantize_gemm_output(const StorageView& c,
                                             const StorageView& a_scale,
                                             const StorageView& b_scale,
                                             const bool transpose_a,
                                             const bool transpose_b,
                                             const StorageView* bias,
                                             StorageView& y) const {
      using DT = sycl_backend::device_type_t<T>;
      const int32_t* input = c.data<int32_t>();
      const float* a = a_scale.data<float>();
      const float* b = b_scale.data<float>();
      const DT* bias_data = bias ? sycl_backend::device_cast(bias->data<T>()) : nullptr;
      DT* output = sycl_backend::device_cast(y.data<T>());
      const dim_t depth = c.dim(-1);
      const dim_t size = c.size();
      const bool a_scalar = a_scale.is_scalar();
      const bool b_scalar = b_scale.is_scalar();
      const bool activate = _activation_type != nullptr;
      const ActivationType activation = activate ? *_activation_type : ActivationType::ReLU;

      sycl_backend::get_queue().parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> id) {
        const dim_t index = id[0];
        const dim_t row = index / depth;
        const dim_t col = index % depth;
        const float scale_a = a[a_scalar ? 0 : (transpose_a ? col : row)];
        const float scale_b = b[b_scalar ? 0 : (transpose_b ? col : row)];
        float value = float(input[index]) / (scale_a * scale_b);
        if (bias_data)
          value += sycl_backend::to_float(bias_data[col]);
        if (activate)
          value = sycl_backend::apply_activation(value, activation);
        output[index] = sycl_backend::from_float<DT>(value);
      });
    }

#define DECLARE_IMPL(T)                                                 \
    template void LayerNorm::compute<Device::SYCL, T>(                 \
      const StorageView*, const StorageView*, const StorageView&, dim_t, dim_t, dim_t, dim_t, StorageView&) const; \
    template void RMSNorm::compute<Device::SYCL, T>(                   \
      const StorageView&, const StorageView&, StorageView&) const;     \
    template void Mean::compute<Device::SYCL, T>(                      \
      const StorageView&, dim_t, dim_t, dim_t, bool, StorageView&) const; \
    template void SoftMax::compute<Device::SYCL, T>(                   \
      const StorageView&, const StorageView*, StorageView&) const;     \
    template void Quantize::quantize<Device::SYCL, T, int8_t>(         \
      const StorageView&, StorageView&, StorageView&) const;           \
    template void Dequantize::dequantize<Device::SYCL, int8_t, T>(     \
      const StorageView&, const StorageView&, StorageView&) const;     \
    template void Dequantize::dequantize_gemm_output<Device::SYCL, T>( \
      const StorageView&, const StorageView&, const StorageView&, bool, bool, const StorageView*, StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

  }
}
