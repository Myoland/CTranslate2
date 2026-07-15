#include "ctranslate2/primitives.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include "ctranslate2/allocator.h"
#include "sycl/helpers.h"
#include "sycl/ops_utils.h"
#include "sycl/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace {

    template <typename T>
    using reduction_type_t = std::conditional_t<
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      float,
      T>;

    template <typename T, typename U>
    T host_convert(const U value) {
      if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>)
        return T(static_cast<float>(value));
      else
        return static_cast<T>(value);
    }

    inline uint32_t round_shift_right_to_even(const uint32_t value,
                                              const unsigned shift) {
      const uint32_t truncated = value >> shift;
      const uint32_t remainder_mask = (uint32_t(1) << shift) - 1;
      const uint32_t remainder = value & remainder_mask;
      const uint32_t halfway = uint32_t(1) << (shift - 1);
      return truncated
        + (remainder > halfway || (remainder == halfway && (truncated & 1)));
    }

    inline float float16_bits_to_float(const uint16_t bits) {
      const uint32_t sign = uint32_t(bits & 0x8000) << 16;
      const uint32_t exponent = (bits >> 10) & 0x1f;
      const uint32_t mantissa = bits & 0x03ff;

      if (exponent == 0) {
        if (mantissa == 0)
          return ::sycl::bit_cast<float>(sign);
        const float magnitude = static_cast<float>(mantissa) * 0x1p-24f;
        return (sign == 0 ? magnitude : -magnitude);
      }

      const uint32_t float_exponent = exponent == 0x1f
        ? 0xff
        : exponent + 112;
      return ::sycl::bit_cast<float>(sign
                                     | (float_exponent << 23)
                                     | (mantissa << 13));
    }

    inline float round_float_to_float16(const float value) {
      const uint32_t bits = ::sycl::bit_cast<uint32_t>(value);
      const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
      const uint32_t exponent_bits = (bits >> 23) & 0xff;
      const uint32_t mantissa = bits & 0x007fffff;

      if (exponent_bits == 0xff) {
        const uint16_t half_mantissa = mantissa == 0
          ? 0
          : static_cast<uint16_t>((mantissa >> 13) | 0x0200);
        return float16_bits_to_float(sign | 0x7c00 | half_mantissa);
      }

      const int exponent = static_cast<int>(exponent_bits) - 127;
      const uint32_t significand = mantissa | 0x00800000;
      uint16_t magnitude = 0;
      if (exponent > 15) {
        magnitude = 0x7c00;
      } else if (exponent >= -14) {
        uint32_t rounded = round_shift_right_to_even(significand, 13);
        int rounded_exponent = exponent;
        if (rounded == 0x0800) {
          rounded = 0x0400;
          ++rounded_exponent;
        }
        if (rounded_exponent > 15) {
          magnitude = 0x7c00;
        } else {
          magnitude = static_cast<uint16_t>(((rounded_exponent + 15) << 10)
                                            | (rounded - 0x0400));
        }
      } else if (exponent >= -25) {
        magnitude = static_cast<uint16_t>(
          round_shift_right_to_even(significand,
                                    static_cast<unsigned>(-exponent - 1)));
      }

      return float16_bits_to_float(sign | magnitude);
    }

    inline float round_float_to_bfloat16(const float value) {
      uint32_t bits = ::sycl::bit_cast<uint32_t>(value);
      const uint32_t exponent = bits & 0x7f800000;
      const uint32_t mantissa = bits & 0x007fffff;
      if (exponent == 0x7f800000) {
        if (mantissa != 0)
          bits |= 0x00400000;
        return ::sycl::bit_cast<float>(bits & 0xffff0000);
      }
      bits += 0x00007fff + ((bits >> 16) & 1);
      return ::sycl::bit_cast<float>(bits & 0xffff0000);
    }

    template <typename T>
    inline float round_to_device_precision(const float value) {
      if constexpr (std::is_same_v<T, ::sycl::half>)
        return round_float_to_float16(value);
      else if constexpr (std::is_same_v<T, ::sycl::ext::oneapi::bfloat16>)
        return round_float_to_bfloat16(value);
      else
        return value;
    }

    // Copies the contiguous innermost rows while swapping dimensions 1 and 2.
    // This is the permutation used to split and combine attention heads.  One
    // work-group owns each row so that the index permutation is computed once
    // per group instead of once per element.
    template <typename T>
    void transpose_0213(const T* input,
                        T* output,
                        const dim_t dim0,
                        const dim_t dim1,
                        const dim_t dim2,
                        const dim_t depth) {
      auto& queue = sycl_backend::get_queue();
      const size_t max_work_group_size
        = queue.get_device().get_info<::sycl::info::device::max_work_group_size>();
      const auto max_work_item_sizes
        = queue.get_device().get_info<::sycl::info::device::max_work_item_sizes<3>>();
      const size_t local_size = std::min({static_cast<size_t>(depth),
                                          size_t(256),
                                          max_work_group_size,
                                          max_work_item_sizes[2]});
      const size_t global_size2 = static_cast<size_t>(dim1) * local_size;

      queue.parallel_for(
        ::sycl::nd_range<3>(
          ::sycl::range<3>(static_cast<size_t>(dim0),
                           static_cast<size_t>(dim2),
                           global_size2),
          ::sycl::range<3>(1, 1, local_size)),
        [=](::sycl::nd_item<3> item) {
          const dim_t i0 = static_cast<dim_t>(item.get_group(0));
          const dim_t i1 = static_cast<dim_t>(item.get_group(2));
          const dim_t i2 = static_cast<dim_t>(item.get_group(1));
          const dim_t input_row = (i0 * dim1 + i1) * dim2 + i2;
          const dim_t output_row = (i0 * dim2 + i2) * dim1 + i1;

          for (dim_t i = static_cast<dim_t>(item.get_local_id(2));
               i < depth;
               i += static_cast<dim_t>(local_size)) {
            output[output_row * depth + i] = input[input_row * depth + i];
          }
        });
    }

    template <typename T, typename Accumulator, typename BinaryOp, typename Transform>
    Accumulator reduce(const T* input,
                       const dim_t size,
                       const Accumulator identity,
                       BinaryOp binary_op,
                       Transform transform) {
      if (size <= 0)
        return identity;

      auto& queue = sycl_backend::get_queue();
      auto* result = ::sycl::malloc_shared<Accumulator>(1, queue);
      if (!result)
        throw std::bad_alloc();
      *result = identity;

      const auto* device_input = sycl_backend::device_cast(input);
      queue.parallel_for(
        ::sycl::range<1>(static_cast<size_t>(size)),
        ::sycl::reduction(result, identity, binary_op),
        [=](::sycl::id<1> id, auto& accumulator) {
          accumulator.combine(transform(device_input[id]));
        });
      queue.wait_and_throw();

      const Accumulator value = *result;
      ::sycl::free(result, queue);
      return value;
    }

    template <typename T, typename UnaryOp>
    void floating_transform(const T* input,
                            T* output,
                            const dim_t size,
                            UnaryOp op) {
      using DeviceT = sycl_backend::device_type_t<T>;
      sycl_backend::unary_transform(input, output, size, [=](DeviceT value) {
        return sycl_backend::device_convert<DeviceT>(op(static_cast<float>(value)));
      });
    }

    template <typename T>
    void scale_matrix(T* output,
                      const dim_t m,
                      const dim_t n,
                      const dim_t ldc,
                      const dim_t stride,
                      const dim_t batch_size,
                      const float beta) {
      if (m <= 0 || n <= 0 || batch_size <= 0)
        return;
      auto* device_output = sycl_backend::device_cast(output);
      const dim_t matrix_size = m * n;
      sycl_backend::get_queue().parallel_for(
        ::sycl::range<1>(static_cast<size_t>(batch_size * matrix_size)),
        [=](::sycl::id<1> id) {
          const dim_t linear = static_cast<dim_t>(id[0]);
          const dim_t batch = linear / matrix_size;
          const dim_t element = linear % matrix_size;
          const dim_t row = element / n;
          const dim_t col = element % n;
          const dim_t offset = batch * stride + row * ldc + col;
          using DeviceT = sycl_backend::device_type_t<T>;
          device_output[offset] = sycl_backend::device_convert<DeviceT>(
            static_cast<float>(device_output[offset]) * beta);
        });
    }

    template <typename T>
    void copy_matrix_to_float(const T* input,
                              float* output,
                              const dim_t m,
                              const dim_t n,
                              const dim_t input_ld,
                              const dim_t input_stride,
                              const dim_t output_ld,
                              const dim_t output_stride,
                              const dim_t batch_size) {
      if (m <= 0 || n <= 0 || batch_size <= 0)
        return;
      const auto* device_input = sycl_backend::device_cast(input);
      const dim_t matrix_size = m * n;
      sycl_backend::get_queue().parallel_for(
        ::sycl::range<1>(static_cast<size_t>(batch_size * matrix_size)),
        [=](::sycl::id<1> id) {
          const dim_t linear = static_cast<dim_t>(id[0]);
          const dim_t batch = linear / matrix_size;
          const dim_t element = linear % matrix_size;
          const dim_t row = element / n;
          const dim_t col = element % n;
          output[batch * output_stride + row * output_ld + col]
            = static_cast<float>(device_input[batch * input_stride + row * input_ld + col]);
        });
    }

    template <typename T>
    void copy_float_to_matrix(const float* input,
                              T* output,
                              const dim_t m,
                              const dim_t n,
                              const dim_t input_ld,
                              const dim_t input_stride,
                              const dim_t output_ld,
                              const dim_t output_stride,
                              const dim_t batch_size) {
      if (m <= 0 || n <= 0 || batch_size <= 0)
        return;
      auto* device_output = sycl_backend::device_cast(output);
      const dim_t matrix_size = m * n;
      sycl_backend::get_queue().parallel_for(
        ::sycl::range<1>(static_cast<size_t>(batch_size * matrix_size)),
        [=](::sycl::id<1> id) {
          const dim_t linear = static_cast<dim_t>(id[0]);
          const dim_t batch = linear / matrix_size;
          const dim_t element = linear % matrix_size;
          const dim_t row = element / n;
          const dim_t col = element % n;
          using DeviceT = sycl_backend::device_type_t<T>;
          device_output[batch * output_stride + row * output_ld + col]
            = sycl_backend::device_convert<DeviceT>(
              input[batch * input_stride + row * input_ld + col]);
        });
    }

    inline oneapi::mkl::transpose to_mkl_transpose(const bool transpose) {
      return transpose ? oneapi::mkl::transpose::trans
                       : oneapi::mkl::transpose::nontrans;
    }

  }

  template <>
  template <typename T>
  T primitives<Device::SYCL>::at(const T* input, const dim_t index) {
    T value{};
    if (index < 0)
      throw std::out_of_range("SYCL scalar index is negative");
    sycl_backend::get_queue().memcpy(&value, input + index, sizeof(T)).wait_and_throw();
    return value;
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::fill(T* output, const T value, const dim_t size) {
    if (size <= 0)
      return;
    auto* device_output = sycl_backend::device_cast(output);
    const auto device_value = sycl_backend::device_scalar(value);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(size)),
      [=](::sycl::id<1> id) { device_output[id] = device_value; });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::strided_fill(T* output,
                                               const T value,
                                               const dim_t stride,
                                               const dim_t size) {
    if (size <= 0)
      return;
    auto* device_output = sycl_backend::device_cast(output);
    const auto device_value = sycl_backend::device_scalar(value);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(size)),
      [=](::sycl::id<1> id) { device_output[id[0] * stride] = device_value; });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::indexed_fill(T* output,
                                               const T value,
                                               const int32_t* indices,
                                               const dim_t num_indices) {
    if (num_indices <= 0)
      return;
    auto* device_output = sycl_backend::device_cast(output);
    const auto device_value = sycl_backend::device_scalar(value);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(num_indices)),
      [=](::sycl::id<1> id) { device_output[indices[id]] = device_value; });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::copy(const T* input, T* output, const dim_t size) {
    if (size <= 0 || input == output)
      return;
    sycl_backend::get_queue().memcpy(output, input, static_cast<size_t>(size) * sizeof(T));
  }

  template <>
  template <typename In, typename Out>
  void primitives<Device::SYCL>::convert(const In* input, Out* output, const dim_t size) {
    if (size <= 0)
      return;
    const auto* device_input = sycl_backend::device_cast(input);
    auto* device_output = sycl_backend::device_cast(output);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(size)),
      [=](::sycl::id<1> id) {
        using DeviceOut = sycl_backend::device_type_t<Out>;
        device_output[id] = sycl_backend::device_convert<DeviceOut>(device_input[id]);
      });
  }

  template void primitives<Device::SYCL>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::SYCL>::convert(const float16_t*, float*, dim_t);
  template void primitives<Device::SYCL>::convert(const float*, bfloat16_t*, dim_t);
  template void primitives<Device::SYCL>::convert(const bfloat16_t*, float*, dim_t);
  template void primitives<Device::SYCL>::convert(const float16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::SYCL>::convert(const bfloat16_t*, float16_t*, dim_t);

  template <>
  template <typename T>
  T primitives<Device::SYCL>::sum(const T* input, const dim_t size) {
    using Acc = reduction_type_t<T>;
    const Acc value = reduce(input,
                             size,
                             Acc(0),
                             ::sycl::plus<Acc>(),
                             [](auto x) { return static_cast<Acc>(x); });
    return host_convert<T>(value);
  }

  template <>
  template <typename T>
  dim_t primitives<Device::SYCL>::max_element(const T* input, const dim_t size) {
    if (size <= 0)
      return 0;
    auto& queue = sycl_backend::get_queue();
    auto* result = ::sycl::malloc_shared<dim_t>(1, queue);
    if (!result)
      throw std::bad_alloc();
    const auto* device_input = sycl_backend::device_cast(input);
    queue.single_task([=]() {
      dim_t best = 0;
      for (dim_t i = 1; i < size; ++i) {
        if (device_input[best] < device_input[i])
          best = i;
      }
      *result = best;
    }).wait_and_throw();
    const dim_t value = *result;
    ::sycl::free(result, queue);
    return value;
  }

  template <>
  template <typename T>
  T primitives<Device::SYCL>::max(const T* input, const dim_t size) {
    using Acc = reduction_type_t<T>;
    if (size <= 0)
      return T();
    const Acc identity = std::numeric_limits<Acc>::lowest();
    const Acc value = reduce(input,
                             size,
                             identity,
                             ::sycl::maximum<Acc>(),
                             [](auto x) { return static_cast<Acc>(x); });
    return host_convert<T>(value);
  }

  template <>
  template <typename T>
  T primitives<Device::SYCL>::amax(const T* input, const dim_t size) {
    using Acc = reduction_type_t<T>;
    const Acc value = reduce(input,
                             size,
                             Acc(0),
                             ::sycl::maximum<Acc>(),
                             [](auto x) {
                               const Acc value = static_cast<Acc>(x);
                               return value < Acc(0) ? -value : value;
                             });
    return host_convert<T>(value);
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::add(const T scalar,
                                      const T* input,
                                      T* output,
                                      const dim_t size) {
    using DeviceT = sycl_backend::device_type_t<T>;
    const DeviceT device_scalar = sycl_backend::device_scalar(scalar);
    sycl_backend::unary_transform(input, output, size, [=](DeviceT value) {
      return sycl_backend::device_convert<DeviceT>(value + device_scalar);
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::add(const T* a,
                                      const T* b,
                                      T* output,
                                      const dim_t size) {
    using DeviceT = sycl_backend::device_type_t<T>;
    sycl_backend::binary_transform(a, b, output, size, [](DeviceT x, DeviceT y) {
      return sycl_backend::device_convert<DeviceT>(x + y);
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::add_batch_broadcast(const T* a,
                                                      const T* b,
                                                      T* output,
                                                      const dim_t a_size,
                                                      const dim_t b_size) {
    if (a_size <= 0 || b_size <= 0)
      return;
    const auto* device_a = sycl_backend::device_cast(a);
    const auto* device_b = sycl_backend::device_cast(b);
    auto* device_output = sycl_backend::device_cast(output);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(b_size)),
      [=](::sycl::id<1> id) { device_output[id] = device_a[id[0] % a_size] + device_b[id]; });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::add_depth_broadcast(const T* a,
                                                      const T* b,
                                                      T* output,
                                                      const dim_t a_size,
                                                      const dim_t b_size) {
    if (a_size <= 0 || b_size <= 0)
      return;
    const dim_t depth = b_size / a_size;
    const auto* device_a = sycl_backend::device_cast(a);
    const auto* device_b = sycl_backend::device_cast(b);
    auto* device_output = sycl_backend::device_cast(output);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(b_size)),
      [=](::sycl::id<1> id) { device_output[id] = device_a[id[0] / depth] + device_b[id]; });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::add_block_broadcast(const T* a,
                                                      const T* b,
                                                      T* output,
                                                      const dim_t block,
                                                      const dim_t a_size,
                                                      const dim_t b_size) {
    if (block <= 0 || a_size <= 0 || b_size <= 0)
      return;
    const auto* device_a = sycl_backend::device_cast(a);
    const auto* device_b = sycl_backend::device_cast(b);
    auto* device_output = sycl_backend::device_cast(output);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(b_size)),
      [=](::sycl::id<1> id) {
        device_output[id] = device_a[(id[0] / block) % a_size] + device_b[id];
      });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::sub(const T* a,
                                      const T* b,
                                      T* output,
                                      const dim_t size) {
    using DeviceT = sycl_backend::device_type_t<T>;
    sycl_backend::binary_transform(a, b, output, size, [](DeviceT x, DeviceT y) {
      return sycl_backend::device_convert<DeviceT>(x - y);
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::min(const T scalar,
                                      const T* input,
                                      T* output,
                                      const dim_t size) {
    using DeviceT = sycl_backend::device_type_t<T>;
    const DeviceT device_scalar = sycl_backend::device_scalar(scalar);
    sycl_backend::unary_transform(input, output, size, [=](DeviceT value) {
      return value < device_scalar ? value : device_scalar;
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::min(const T* a,
                                      const T* b,
                                      T* output,
                                      const dim_t size) {
    using DeviceT = sycl_backend::device_type_t<T>;
    sycl_backend::binary_transform(a, b, output, size, [](DeviceT x, DeviceT y) {
      return x < y ? x : y;
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::max(const T scalar,
                                      const T* input,
                                      T* output,
                                      const dim_t size) {
    using DeviceT = sycl_backend::device_type_t<T>;
    const DeviceT device_scalar = sycl_backend::device_scalar(scalar);
    sycl_backend::unary_transform(input, output, size, [=](DeviceT value) {
      return value > device_scalar ? value : device_scalar;
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::max(const T* a,
                                      const T* b,
                                      T* output,
                                      const dim_t size) {
    using DeviceT = sycl_backend::device_type_t<T>;
    sycl_backend::binary_transform(a, b, output, size, [](DeviceT x, DeviceT y) {
      return x > y ? x : y;
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::mul(const T scalar,
                                      const T* input,
                                      T* output,
                                      const dim_t size) {
    using DeviceT = sycl_backend::device_type_t<T>;
    const DeviceT device_scalar = sycl_backend::device_scalar(scalar);
    sycl_backend::unary_transform(input, output, size, [=](DeviceT value) {
      return sycl_backend::device_convert<DeviceT>(value * device_scalar);
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::mul(const T* a,
                                      const T* b,
                                      T* output,
                                      const dim_t size) {
    using DeviceT = sycl_backend::device_type_t<T>;
    sycl_backend::binary_transform(a, b, output, size, [](DeviceT x, DeviceT y) {
      return sycl_backend::device_convert<DeviceT>(x * y);
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::mul_batch_broadcast(const T* a,
                                                      const T* b,
                                                      T* output,
                                                      const dim_t a_size,
                                                      const dim_t b_size) {
    if (a_size <= 0 || b_size <= 0)
      return;
    const auto* device_a = sycl_backend::device_cast(a);
    const auto* device_b = sycl_backend::device_cast(b);
    auto* device_output = sycl_backend::device_cast(output);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(b_size)),
      [=](::sycl::id<1> id) { device_output[id] = device_a[id[0] % a_size] * device_b[id]; });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::penalize_previous_tokens(T* scores,
                                                           const T* previous_scores,
                                                           const int32_t* previous_ids,
                                                           const T penalty,
                                                           const dim_t batch_size,
                                                           const dim_t length,
                                                           const dim_t vocabulary_size) {
    if (batch_size <= 0 || length <= 0)
      return;
    auto* device_scores = sycl_backend::device_cast(scores);
    const auto* device_previous_scores = sycl_backend::device_cast(previous_scores);
    const float penalty_value = static_cast<float>(penalty);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(batch_size * length)),
      [=](::sycl::id<1> id) {
        const dim_t i = static_cast<dim_t>(id[0]);
        const dim_t write_index = (i / length) * vocabulary_size + previous_ids[i];
        const float score = static_cast<float>(device_previous_scores[i]);
        using DeviceT = sycl_backend::device_type_t<T>;
        device_scores[write_index] = sycl_backend::device_convert<DeviceT>(
          score < 0.f ? score * penalty_value : score / penalty_value);
      });
  }

  template <>
  void primitives<Device::SYCL>::prepare_length_mask(const int32_t* lengths,
                                                      const dim_t batch_size,
                                                      const dim_t num_heads,
                                                      const dim_t num_queries,
                                                      const bool mask_future,
                                                      const bool multi_query,
                                                      int32_t* mask) {
    if (batch_size <= 0 || num_heads <= 0 || num_queries <= 0)
      return;
    const dim_t batch_stride = num_heads * num_queries;
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(batch_size * batch_stride)),
      [=](::sycl::id<1> id) {
        const dim_t i = static_cast<dim_t>(id[0]);
        const dim_t batch = i / batch_stride;
        const dim_t query_head_index = i % batch_stride;
        const int32_t length = lengths[batch];
        if (!mask_future) {
          mask[i] = length;
          return;
        }
        const dim_t query = multi_query
          ? query_head_index / num_heads
          : query_head_index % num_queries;
        const int32_t future_length = static_cast<int32_t>(query + 1);
        mask[i] = length < future_length ? length : future_length;
      });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::transpose_2d(const T* input,
                                               const dim_t* dims,
                                               T* output) {
    const dim_t rows = dims[0];
    const dim_t cols = dims[1];
    if (rows <= 0 || cols <= 0)
      return;
    const auto* device_input = sycl_backend::device_cast(input);
    auto* device_output = sycl_backend::device_cast(output);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(rows * cols)),
      [=](::sycl::id<1> id) {
        const dim_t i = static_cast<dim_t>(id[0]);
        device_output[i] = device_input[(i % rows) * cols + i / rows];
      });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::transpose_3d(const T* input,
                                               const dim_t* dims,
                                               const dim_t* perm,
                                               T* output) {
    const dim_t a_stride0 = dims[1] * dims[2];
    const dim_t a_stride1 = dims[2];
    const dim_t a_strides[3] = {a_stride0, a_stride1, 1};
    const dim_t input_stride0 = a_strides[perm[0]];
    const dim_t input_stride1 = a_strides[perm[1]];
    const dim_t input_stride2 = a_strides[perm[2]];
    const dim_t output_dim1 = dims[perm[1]];
    const dim_t output_dim2 = dims[perm[2]];
    const dim_t output_stride0 = output_dim1 * output_dim2;
    const dim_t size = dims[0] * dims[1] * dims[2];
    if (size <= 0)
      return;
    const auto* device_input = sycl_backend::device_cast(input);
    auto* device_output = sycl_backend::device_cast(output);
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(size)),
      [=](::sycl::id<1> id) {
        const dim_t i = static_cast<dim_t>(id[0]);
        const dim_t i0 = i / output_stride0;
        const dim_t i1 = i / output_dim2 % output_dim1;
        const dim_t i2 = i % output_dim2;
        device_output[i] = device_input[i0 * input_stride0
                                        + i1 * input_stride1
                                        + i2 * input_stride2];
      });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::transpose_4d(const T* input,
                                               const dim_t* dims,
                                               const dim_t* perm,
                                               T* output) {
    const dim_t size = dims[0] * dims[1] * dims[2] * dims[3];
    if (size <= 0)
      return;

    const auto* device_input = sycl_backend::device_cast(input);
    auto* device_output = sycl_backend::device_cast(output);

    if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3) {
      // Optimize the permutation used in multi-head attention.  The 16-byte
      // copy preserves the underlying bits, including floating-point NaNs.
      using DeviceT = sycl_backend::device_type_t<T>;
      using Copy16 = ::sycl::vec<uint32_t, 4>;
      static_assert(sizeof(Copy16) == 16, "Copy16 must be 16 bytes");
      static_assert(alignof(Copy16) == 16, "Copy16 must be 16-byte aligned");
      static_assert(sizeof(Copy16) % sizeof(DeviceT) == 0,
                    "Device element size must divide Copy16");

      constexpr dim_t elements_per_copy = sizeof(Copy16) / sizeof(DeviceT);
      const bool aligned
        = reinterpret_cast<uintptr_t>(device_input) % alignof(Copy16) == 0
          && reinterpret_cast<uintptr_t>(device_output) % alignof(Copy16) == 0;
      if (aligned && dims[3] % elements_per_copy == 0) {
        transpose_0213(reinterpret_cast<const Copy16*>(device_input),
                       reinterpret_cast<Copy16*>(device_output),
                       dims[0],
                       dims[1],
                       dims[2],
                       dims[3] / elements_per_copy);
      } else {
        transpose_0213(device_input,
                       device_output,
                       dims[0],
                       dims[1],
                       dims[2],
                       dims[3]);
      }
      return;
    }

    const dim_t a_strides[4] = {
      dims[1] * dims[2] * dims[3],
      dims[2] * dims[3],
      dims[3],
      1
    };
    const dim_t input_stride0 = a_strides[perm[0]];
    const dim_t input_stride1 = a_strides[perm[1]];
    const dim_t input_stride2 = a_strides[perm[2]];
    const dim_t input_stride3 = a_strides[perm[3]];
    const dim_t output_dim1 = dims[perm[1]];
    const dim_t output_dim2 = dims[perm[2]];
    const dim_t output_dim3 = dims[perm[3]];
    const dim_t output_stride0 = output_dim1 * output_dim2 * output_dim3;
    const dim_t output_stride1 = output_dim2 * output_dim3;
    sycl_backend::get_queue().parallel_for(
      ::sycl::range<1>(static_cast<size_t>(size)),
      [=](::sycl::id<1> id) {
        const dim_t i = static_cast<dim_t>(id[0]);
        const dim_t i0 = i / output_stride0;
        const dim_t i1 = i / output_stride1 % output_dim1;
        const dim_t i2 = i / output_dim3 % output_dim2;
        const dim_t i3 = i % output_dim3;
        device_output[i] = device_input[i0 * input_stride0
                                        + i1 * input_stride1
                                        + i2 * input_stride2
                                        + i3 * input_stride3];
      });
  }

  template <>
  template <typename T>
  float primitives<Device::SYCL>::logsumexp(const T* input, const dim_t size) {
    if (size <= 0)
      return -std::numeric_limits<float>::infinity();
    const float max_value = static_cast<float>(max(input, size));
    const float exp_sum = reduce(input,
                                 size,
                                 0.f,
                                 ::sycl::plus<float>(),
                                 [=](auto value) {
                                   return ::sycl::exp(static_cast<float>(value) - max_value);
                                 });
    return std::log(exp_sum) + max_value;
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::suppress_text_if_timestamp(
    T* logits,
    const T* scores,
    const dim_t rows,
    const dim_t depth,
    const dim_t num_text_tokens,
    const dim_t timestamp_begin,
    const dim_t num_timestamp_tokens,
    const uint64_t check_rows,
    const T disable_value) {
    if (rows > 64)
      throw std::invalid_argument("Whisper timestamp row mask supports at most 64 rows");
    if (depth < 0 || num_text_tokens < 0 || num_text_tokens > depth
        || timestamp_begin < 0 || timestamp_begin > depth
        || num_timestamp_tokens < 0
        || num_timestamp_tokens > depth - timestamp_begin) {
      throw std::invalid_argument("Whisper timestamp ranges are outside the score depth");
    }
    if (rows <= 0 || depth <= 0 || num_text_tokens <= 0
        || num_timestamp_tokens <= 0 || check_rows == 0)
      return;

    auto& queue = sycl_backend::get_queue();
    const size_t local_size = sycl_backend::work_group_size();
    using DeviceT = sycl_backend::device_type_t<T>;
    auto* device_logits = sycl_backend::device_cast(logits);
    const auto* device_scores = sycl_backend::device_cast(scores);
    const DeviceT device_disable_value = sycl_backend::device_scalar(disable_value);
    const float lowest = std::numeric_limits<float>::lowest();
    const float negative_infinity = -std::numeric_limits<float>::infinity();

    queue.submit([&](::sycl::handler& handler) {
      ::sycl::local_accessor<float, 1> scratch(local_size, handler);
      handler.parallel_for(
        ::sycl::nd_range<1>(::sycl::range<1>(static_cast<size_t>(rows) * local_size),
                            ::sycl::range<1>(local_size)),
        [=](::sycl::nd_item<1> item) {
        const dim_t row = static_cast<dim_t>(item.get_group_linear_id());
        if ((check_rows & (uint64_t(1) << row)) == 0)
          return;

        const dim_t local_id = static_cast<dim_t>(item.get_local_linear_id());
        const dim_t group_size = static_cast<dim_t>(item.get_local_range(0));
        const auto group = item.get_group();
        const auto* row_scores = device_scores + row * depth;

        // Reproduce the low-precision values produced by LogSoftMax before
        // evaluating the timestamp rule. Although the normalization constant
        // cancels algebraically in exact arithmetic, rounding each normalized
        // score to FLOAT16 or BFLOAT16 can change a boundary decision.
        float normalization_max = negative_infinity;
        for (dim_t i = local_id; i < depth; i += group_size)
          normalization_max = ::sycl::fmax(
            normalization_max,
            sycl_backend::to_float(row_scores[i]));
        scratch[local_id] = normalization_max;
        item.barrier(::sycl::access::fence_space::local_space);
        for (dim_t stride = group_size / 2; stride > 0; stride >>= 1) {
          if (local_id < stride) {
            scratch[local_id] = ::sycl::fmax(scratch[local_id],
                                             scratch[local_id + stride]);
          }
          item.barrier(::sycl::access::fence_space::local_space);
        }
        normalization_max = scratch[0];
        item.barrier(::sycl::access::fence_space::local_space);

        float row_exp_sum = 0;
        for (dim_t i = local_id; i < depth; i += group_size) {
          row_exp_sum += ::sycl::exp(sycl_backend::to_float(row_scores[i])
                                     - normalization_max);
        }
        scratch[local_id] = row_exp_sum;
        item.barrier(::sycl::access::fence_space::local_space);
        for (dim_t stride = group_size / 2; stride > 0; stride >>= 1) {
          if (local_id < stride)
            scratch[local_id] += scratch[local_id + stride];
          item.barrier(::sycl::access::fence_space::local_space);
        }
        const float log_denominator = ::sycl::log(scratch[0]);
        item.barrier(::sycl::access::fence_space::local_space);
        const auto normalized_score = [=](const DeviceT score) {
          return round_to_device_precision<DeviceT>(
            (sycl_backend::to_float(score) - normalization_max)
            - log_denominator);
        };

        float text_max = lowest;
        for (dim_t i = local_id; i < num_text_tokens; i += group_size)
          text_max = ::sycl::fmax(text_max,
                                  normalized_score(row_scores[i]));
        text_max = ::sycl::reduce_over_group(group,
                                             text_max,
                                             ::sycl::maximum<float>());

        float timestamp_max = lowest;
        for (dim_t i = local_id; i < num_timestamp_tokens; i += group_size) {
          timestamp_max = ::sycl::fmax(
            timestamp_max,
            normalized_score(row_scores[timestamp_begin + i]));
        }
        timestamp_max = ::sycl::reduce_over_group(group,
                                                  timestamp_max,
                                                  ::sycl::maximum<float>());

        float timestamp_exp_sum = 0;
        for (dim_t i = local_id; i < num_timestamp_tokens; i += group_size) {
          timestamp_exp_sum += ::sycl::exp(
            normalized_score(row_scores[timestamp_begin + i]) - timestamp_max);
        }
        scratch[local_id] = timestamp_exp_sum;
        item.barrier(::sycl::access::fence_space::local_space);
        for (dim_t stride = group_size / 2; stride > 0; stride >>= 1) {
          if (local_id < stride)
            scratch[local_id] += scratch[local_id + stride];
          item.barrier(::sycl::access::fence_space::local_space);
        }
        timestamp_exp_sum = scratch[0];

        if (::sycl::log(timestamp_exp_sum) + timestamp_max > text_max) {
          auto* row_logits = device_logits + row * depth;
          for (dim_t i = local_id; i < num_text_tokens; i += group_size)
            row_logits[i] = device_disable_value;
        }
        });
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::exp(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) { return ::sycl::exp(value); });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::log(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) { return ::sycl::log(value); });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::cos(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) { return ::sycl::cos(value); });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::sin(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) { return ::sycl::sin(value); });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::tanh(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) { return ::sycl::tanh(value); });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::relu(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) { return value > 0.f ? value : 0.f; });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::gelu(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) {
      return .5f * value * (1.f + ::sycl::erf(.7071067811865475f * value));
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::gelu_tanh(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) {
      return .5f * value
        * (1.f + ::sycl::tanh(.7978845608028654f
                              * (value + .044715f * value * value * value)));
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::gelu_sigmoid(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) {
      return value / (1.f + ::sycl::exp(-1.702f * value));
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::sigmoid(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) {
      return 1.f / (1.f + ::sycl::exp(-value));
    });
  }

  template <>
  template <typename T>
  void primitives<Device::SYCL>::swish(const T* input, T* output, const dim_t size) {
    floating_transform(input, output, size, [](float value) {
      return value / (1.f + ::sycl::exp(-value));
    });
  }

  namespace {

    template <typename In, typename Out, typename Scalar>
    void submit_gemm(const bool transpose_a,
                     const bool transpose_b,
                     const dim_t m,
                     const dim_t n,
                     const dim_t k,
                     const Scalar alpha,
                     const In* a,
                     const dim_t lda,
                     const In* b,
                     const dim_t ldb,
                     const Scalar beta,
                     Out* c,
                     const dim_t ldc) {
      oneapi::mkl::blas::row_major::gemm(sycl_backend::get_queue(),
                                          to_mkl_transpose(transpose_a),
                                          to_mkl_transpose(transpose_b),
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          a,
                                          lda,
                                          b,
                                          ldb,
                                          beta,
                                          c,
                                          ldc);
    }

    template <typename In, typename Out, typename Scalar>
    void submit_gemm_batch(const bool transpose_a,
                           const bool transpose_b,
                           const dim_t m,
                           const dim_t n,
                           const dim_t k,
                           const Scalar alpha,
                           const In* a,
                           const dim_t lda,
                           const dim_t stridea,
                           const In* b,
                           const dim_t ldb,
                           const dim_t strideb,
                           const Scalar beta,
                           Out* c,
                           const dim_t ldc,
                           const dim_t stridec,
                           const dim_t batch_size) {
      // A zero input stride is used by grouped Conv1D to broadcast its weights.
      // oneMKL documents a full-matrix minimum stride, so submit individual
      // GEMMs for broadcast batches instead of depending on an extension.
      if (stridea == 0 || strideb == 0) {
        for (dim_t batch = 0; batch < batch_size; ++batch) {
          submit_gemm(transpose_a,
                      transpose_b,
                      m,
                      n,
                      k,
                      alpha,
                      a + batch * stridea,
                      lda,
                      b + batch * strideb,
                      ldb,
                      beta,
                      c + batch * stridec,
                      ldc);
        }
        return;
      }

      oneapi::mkl::blas::row_major::gemm_batch(sycl_backend::get_queue(),
                                                to_mkl_transpose(transpose_a),
                                                to_mkl_transpose(transpose_b),
                                                m,
                                                n,
                                                k,
                                                alpha,
                                                a,
                                                lda,
                                                stridea,
                                                b,
                                                ldb,
                                                strideb,
                                                beta,
                                                c,
                                                ldc,
                                                stridec,
                                                batch_size);
    }

    template <typename T>
    void low_precision_gemm(const bool transpose_a,
                            const bool transpose_b,
                            const dim_t m,
                            const dim_t n,
                            const dim_t k,
                            const float alpha,
                            const T* a,
                            const dim_t lda,
                            const dim_t stridea,
                            const T* b,
                            const dim_t ldb,
                            const dim_t strideb,
                            const float beta,
                            T* c,
                            const dim_t ldc,
                            const dim_t stridec,
                            const dim_t batch_size) {
      if (m <= 0 || n <= 0 || batch_size <= 0)
        return;
      if (k <= 0 || alpha == 0.f) {
        scale_matrix(c, m, n, ldc, stridec, batch_size, beta);
        return;
      }

      if constexpr (std::is_same_v<T, float16_t>) {
        // The native half-output oneMKL overload avoids an FP32 output
        // workspace and the two conversion kernels.  That overload also takes
        // half-precision alpha and beta, so only use it when converting these
        // scalars is lossless.  Preserve the mixed-precision path below for
        // arbitrary scaling factors.
        const ::sycl::half half_alpha(alpha);
        const ::sycl::half half_beta(beta);
        if (std::isfinite(alpha)
            && std::isfinite(beta)
            && static_cast<float>(half_alpha) == alpha
            && static_cast<float>(half_beta) == beta) {
          const auto* device_a = sycl_backend::device_cast(a);
          const auto* device_b = sycl_backend::device_cast(b);
          auto* device_c = sycl_backend::device_cast(c);
          if (batch_size == 1) {
            submit_gemm(transpose_a,
                        transpose_b,
                        m,
                        n,
                        k,
                        half_alpha,
                        device_a,
                        lda,
                        device_b,
                        ldb,
                        half_beta,
                        device_c,
                        ldc);
          } else {
            submit_gemm_batch(transpose_a,
                              transpose_b,
                              m,
                              n,
                              k,
                              half_alpha,
                              device_a,
                              lda,
                              stridea,
                              device_b,
                              ldb,
                              strideb,
                              half_beta,
                              device_c,
                              ldc,
                              stridec,
                              batch_size);
          }
          return;
        }
      }

      const dim_t matrix_span = (m - 1) * ldc + n;
      const dim_t temp_stride = batch_size == 1 ? matrix_span : stridec;
      const dim_t temp_size = (batch_size - 1) * temp_stride + matrix_span;
      Allocator& allocator = get_allocator<Device::SYCL>();
      auto* temp = static_cast<float*>(allocator.allocate(
        static_cast<size_t>(temp_size) * sizeof(float)));

      try {
        if (beta != 0.f) {
          copy_matrix_to_float(c,
                               temp,
                               m,
                               n,
                               ldc,
                               stridec,
                               ldc,
                               temp_stride,
                               batch_size);
        }

        const auto* device_a = sycl_backend::device_cast(a);
        const auto* device_b = sycl_backend::device_cast(b);
        if (batch_size == 1) {
          submit_gemm(transpose_a,
                      transpose_b,
                      m,
                      n,
                      k,
                      alpha,
                      device_a,
                      lda,
                      device_b,
                      ldb,
                      beta,
                      temp,
                      ldc);
        } else {
          submit_gemm_batch(transpose_a,
                            transpose_b,
                            m,
                            n,
                            k,
                            alpha,
                            device_a,
                            lda,
                            stridea,
                            device_b,
                            ldb,
                            strideb,
                            beta,
                            temp,
                            ldc,
                            temp_stride,
                            batch_size);
        }

        copy_float_to_matrix(temp,
                             c,
                             m,
                             n,
                             ldc,
                             temp_stride,
                             ldc,
                             stridec,
                             batch_size);
      } catch (...) {
        allocator.free(temp);
        throw;
      }
      allocator.free(temp);
    }

    inline void check_unpacked(const bool a_is_packed, const bool b_is_packed) {
      if (a_is_packed || b_is_packed)
        throw std::invalid_argument("Packed GEMM is not supported by the SYCL backend");
    }

  }

  template <>
  template <>
  void primitives<Device::SYCL>::gemm(bool a_is_packed,
                                       bool b_is_packed,
                                       bool transpose_a,
                                       bool transpose_b,
                                       dim_t m,
                                       dim_t n,
                                       dim_t k,
                                       float alpha,
                                       const float* a,
                                       dim_t lda,
                                       const float* b,
                                       dim_t ldb,
                                       float beta,
                                       float* c,
                                       dim_t ldc,
                                       const float*) {
    check_unpacked(a_is_packed, b_is_packed);
    if (m <= 0 || n <= 0)
      return;
    if (k <= 0 || alpha == 0.f) {
      scale_matrix(c, m, n, ldc, m * ldc, 1, beta);
      return;
    }
    submit_gemm(transpose_a, transpose_b,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  template <>
  template <>
  void primitives<Device::SYCL>::gemm(bool a_is_packed,
                                       bool b_is_packed,
                                       bool transpose_a,
                                       bool transpose_b,
                                       dim_t m,
                                       dim_t n,
                                       dim_t k,
                                       float alpha,
                                       const float16_t* a,
                                       dim_t lda,
                                       const float16_t* b,
                                       dim_t ldb,
                                       float beta,
                                       float16_t* c,
                                       dim_t ldc,
                                       const float16_t*) {
    check_unpacked(a_is_packed, b_is_packed);
    low_precision_gemm(transpose_a, transpose_b,
                       m, n, k, alpha,
                       a, lda, 0,
                       b, ldb, 0,
                       beta, c, ldc, m * ldc, 1);
  }

  template <>
  template <>
  void primitives<Device::SYCL>::gemm(bool a_is_packed,
                                       bool b_is_packed,
                                       bool transpose_a,
                                       bool transpose_b,
                                       dim_t m,
                                       dim_t n,
                                       dim_t k,
                                       float alpha,
                                       const bfloat16_t* a,
                                       dim_t lda,
                                       const bfloat16_t* b,
                                       dim_t ldb,
                                       float beta,
                                       bfloat16_t* c,
                                       dim_t ldc,
                                       const bfloat16_t*) {
    check_unpacked(a_is_packed, b_is_packed);
    low_precision_gemm(transpose_a, transpose_b,
                       m, n, k, alpha,
                       a, lda, 0,
                       b, ldb, 0,
                       beta, c, ldc, m * ldc, 1);
  }

  template <>
  template <>
  void primitives<Device::SYCL>::gemm(bool a_is_packed,
                                       bool b_is_packed,
                                       bool transpose_a,
                                       bool transpose_b,
                                       dim_t m,
                                       dim_t n,
                                       dim_t k,
                                       float alpha,
                                       const int8_t* a,
                                       dim_t lda,
                                       const int8_t* b,
                                       dim_t ldb,
                                       float beta,
                                       int32_t* c,
                                       dim_t ldc,
                                       const int32_t*) {
    check_unpacked(a_is_packed, b_is_packed);
    if (m <= 0 || n <= 0)
      return;
    if (k <= 0 || alpha == 0.f) {
      scale_matrix(c, m, n, ldc, m * ldc, 1, beta);
      return;
    }
    submit_gemm(transpose_a, transpose_b,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  template <>
  template <>
  void primitives<Device::SYCL>::gemm_batch_strided(bool transpose_a,
                                                     bool transpose_b,
                                                     dim_t m,
                                                     dim_t n,
                                                     dim_t k,
                                                     float alpha,
                                                     const float* a,
                                                     dim_t lda,
                                                     dim_t stridea,
                                                     const float* b,
                                                     dim_t ldb,
                                                     dim_t strideb,
                                                     float beta,
                                                     float* c,
                                                     dim_t ldc,
                                                     dim_t stridec,
                                                     dim_t batch_size) {
    if (m <= 0 || n <= 0 || batch_size <= 0)
      return;
    if (k <= 0 || alpha == 0.f) {
      scale_matrix(c, m, n, ldc, stridec, batch_size, beta);
      return;
    }
    submit_gemm_batch(transpose_a, transpose_b,
                      m, n, k, alpha,
                      a, lda, stridea,
                      b, ldb, strideb,
                      beta, c, ldc, stridec, batch_size);
  }

  template <>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::SYCL>::copy(const T* input,
                                                                 T* output,
                                                                 const dim_t size) {
    if (size <= 0)
      return;
    sycl_backend::get_queue()
      .memcpy(output, input, static_cast<size_t>(size) * sizeof(T))
      .wait_and_throw();
  }

  template <>
  template <typename T>
  void cross_device_primitives<Device::SYCL, Device::CPU>::copy(const T* input,
                                                                 T* output,
                                                                 const dim_t size) {
    if (size <= 0)
      return;
    sycl_backend::get_queue()
      .memcpy(output, input, static_cast<size_t>(size) * sizeof(T))
      .wait_and_throw();
  }

#define DECLARE_IMPL(T)                                                  \
  template T primitives<Device::SYCL>::at(const T*, dim_t);             \
  template void primitives<Device::SYCL>::fill(T*, T, dim_t);           \
  template void primitives<Device::SYCL>::strided_fill(T*, T, dim_t, dim_t); \
  template void primitives<Device::SYCL>::indexed_fill(T*, T, const int32_t*, dim_t); \
  template void primitives<Device::SYCL>::copy(const T*, T*, dim_t);     \
  template T primitives<Device::SYCL>::sum(const T*, dim_t);             \
  template dim_t primitives<Device::SYCL>::max_element(const T*, dim_t); \
  template T primitives<Device::SYCL>::max(const T*, dim_t);             \
  template T primitives<Device::SYCL>::amax(const T*, dim_t);            \
  template void primitives<Device::SYCL>::add(T, const T*, T*, dim_t);   \
  template void primitives<Device::SYCL>::add(const T*, const T*, T*, dim_t); \
  template void primitives<Device::SYCL>::add_batch_broadcast(           \
    const T*, const T*, T*, dim_t, dim_t);                               \
  template void primitives<Device::SYCL>::add_depth_broadcast(           \
    const T*, const T*, T*, dim_t, dim_t);                               \
  template void primitives<Device::SYCL>::add_block_broadcast(           \
    const T*, const T*, T*, dim_t, dim_t, dim_t);                        \
  template void primitives<Device::SYCL>::sub(const T*, const T*, T*, dim_t); \
  template void primitives<Device::SYCL>::min(T, const T*, T*, dim_t);   \
  template void primitives<Device::SYCL>::min(const T*, const T*, T*, dim_t); \
  template void primitives<Device::SYCL>::max(T, const T*, T*, dim_t);   \
  template void primitives<Device::SYCL>::max(const T*, const T*, T*, dim_t); \
  template void primitives<Device::SYCL>::mul(T, const T*, T*, dim_t);   \
  template void primitives<Device::SYCL>::mul(const T*, const T*, T*, dim_t); \
  template void primitives<Device::SYCL>::mul_batch_broadcast(           \
    const T*, const T*, T*, dim_t, dim_t);                               \
  template void primitives<Device::SYCL>::penalize_previous_tokens(      \
    T*, const T*, const int32_t*, T, dim_t, dim_t, dim_t);               \
  template void primitives<Device::SYCL>::transpose_2d(                  \
    const T*, const dim_t*, T*);                                         \
  template void primitives<Device::SYCL>::transpose_3d(                  \
    const T*, const dim_t*, const dim_t*, T*);                           \
  template void primitives<Device::SYCL>::transpose_4d(                  \
    const T*, const dim_t*, const dim_t*, T*);                           \
  template void cross_device_primitives<Device::CPU, Device::SYCL>::copy(\
    const T*, T*, dim_t);                                                \
  template void cross_device_primitives<Device::SYCL, Device::CPU>::copy(\
    const T*, T*, dim_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#undef DECLARE_IMPL

#define DECLARE_FLOAT_IMPL(T)                                            \
  template void primitives<Device::SYCL>::relu(const T*, T*, dim_t);     \
  template void primitives<Device::SYCL>::gelu(const T*, T*, dim_t);     \
  template void primitives<Device::SYCL>::gelu_tanh(const T*, T*, dim_t); \
  template void primitives<Device::SYCL>::gelu_sigmoid(const T*, T*, dim_t); \
  template void primitives<Device::SYCL>::sigmoid(const T*, T*, dim_t);  \
  template void primitives<Device::SYCL>::swish(const T*, T*, dim_t);    \
  template float primitives<Device::SYCL>::logsumexp(const T*, dim_t);   \
  template void primitives<Device::SYCL>::sin(const T*, T*, dim_t);      \
  template void primitives<Device::SYCL>::cos(const T*, T*, dim_t);      \
  template void primitives<Device::SYCL>::tanh(const T*, T*, dim_t);     \
  template void primitives<Device::SYCL>::exp(const T*, T*, dim_t);      \
  template void primitives<Device::SYCL>::log(const T*, T*, dim_t);

#define DECLARE_WHISPER_TIMESTAMP_IMPL(T)                                \
  template void primitives<Device::SYCL>::suppress_text_if_timestamp(    \
    T*, const T*, dim_t, dim_t, dim_t, dim_t, dim_t, uint64_t, T);

  DECLARE_FLOAT_IMPL(float)
  DECLARE_FLOAT_IMPL(float16_t)
  DECLARE_FLOAT_IMPL(bfloat16_t)

  DECLARE_WHISPER_TIMESTAMP_IMPL(float)
  DECLARE_WHISPER_TIMESTAMP_IMPL(float16_t)
  DECLARE_WHISPER_TIMESTAMP_IMPL(bfloat16_t)

#undef DECLARE_FLOAT_IMPL
#undef DECLARE_WHISPER_TIMESTAMP_IMPL

}

namespace ctranslate2 {

  template <>
  template <>
  void primitives<Device::SYCL>::gemm_batch_strided(bool transpose_a,
                                                     bool transpose_b,
                                                     dim_t m,
                                                     dim_t n,
                                                     dim_t k,
                                                     float alpha,
                                                     const float16_t* a,
                                                     dim_t lda,
                                                     dim_t stridea,
                                                     const float16_t* b,
                                                     dim_t ldb,
                                                     dim_t strideb,
                                                     float beta,
                                                     float16_t* c,
                                                     dim_t ldc,
                                                     dim_t stridec,
                                                     dim_t batch_size) {
    low_precision_gemm(transpose_a, transpose_b,
                       m, n, k, alpha,
                       a, lda, stridea,
                       b, ldb, strideb,
                       beta, c, ldc, stridec, batch_size);
  }

  template <>
  template <>
  void primitives<Device::SYCL>::gemm_batch_strided(bool transpose_a,
                                                     bool transpose_b,
                                                     dim_t m,
                                                     dim_t n,
                                                     dim_t k,
                                                     float alpha,
                                                     const bfloat16_t* a,
                                                     dim_t lda,
                                                     dim_t stridea,
                                                     const bfloat16_t* b,
                                                     dim_t ldb,
                                                     dim_t strideb,
                                                     float beta,
                                                     bfloat16_t* c,
                                                     dim_t ldc,
                                                     dim_t stridec,
                                                     dim_t batch_size) {
    low_precision_gemm(transpose_a, transpose_b,
                       m, n, k, alpha,
                       a, lda, stridea,
                       b, ldb, strideb,
                       beta, c, ldc, stridec, batch_size);
  }

  template <>
  template <>
  void primitives<Device::SYCL>::gemm_batch_strided(bool transpose_a,
                                                     bool transpose_b,
                                                     dim_t m,
                                                     dim_t n,
                                                     dim_t k,
                                                     float alpha,
                                                     const int8_t* a,
                                                     dim_t lda,
                                                     dim_t stridea,
                                                     const int8_t* b,
                                                     dim_t ldb,
                                                     dim_t strideb,
                                                     float beta,
                                                     int32_t* c,
                                                     dim_t ldc,
                                                     dim_t stridec,
                                                     dim_t batch_size) {
    if (m <= 0 || n <= 0 || batch_size <= 0)
      return;
    if (k <= 0 || alpha == 0.f) {
      scale_matrix(c, m, n, ldc, stridec, batch_size, beta);
      return;
    }
    submit_gemm_batch(transpose_a, transpose_b,
                      m, n, k, alpha,
                      a, lda, stridea,
                      b, ldb, strideb,
                      beta, c, ldc, stridec, batch_size);
  }

}
