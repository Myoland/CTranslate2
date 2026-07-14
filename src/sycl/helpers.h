#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include "ctranslate2/types.h"
#include "sycl/utils.h"

namespace ctranslate2 {
  namespace sycl_backend {

    // float16_t and bfloat16_t are host-side storage types.  Their bit layouts
    // match the native SYCL types, but their conversion operators are not
    // device-callable.  Always reinterpret them at the kernel boundary.
    template <typename T>
    struct device_type {
      using type = T;
    };

    template <>
    struct device_type<float16_t> {
      using type = ::sycl::half;
    };

    template <>
    struct device_type<bfloat16_t> {
      using type = ::sycl::ext::oneapi::bfloat16;
    };

    template <typename T>
    using device_type_t = typename device_type<T>::type;

    static_assert(sizeof(float16_t) == sizeof(::sycl::half),
                  "CTranslate2 and SYCL float16 storage must have the same size");
    static_assert(sizeof(bfloat16_t) == sizeof(::sycl::ext::oneapi::bfloat16),
                  "CTranslate2 and SYCL bfloat16 storage must have the same size");

    template <typename T>
    inline device_type_t<T>* device_cast(T* pointer) {
      return reinterpret_cast<device_type_t<T>*>(pointer);
    }

    template <typename T>
    inline const device_type_t<T>* device_cast(const T* pointer) {
      return reinterpret_cast<const device_type_t<T>*>(pointer);
    }

    template <typename T>
    inline device_type_t<T> device_scalar(const T value) {
      if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>)
        return device_type_t<T>(static_cast<float>(value));
      else
        return value;
    }

    template <typename T>
    inline float to_float(const T value) {
      return static_cast<float>(value);
    }

    template <typename Out, typename In>
    inline Out device_convert(const In value) {
      if constexpr (std::is_same_v<Out, ::sycl::half>
                    || std::is_same_v<Out, ::sycl::ext::oneapi::bfloat16>)
        return Out(static_cast<float>(value));
      else
        return static_cast<Out>(value);
    }

    template <typename T, typename UnaryOp>
    inline void unary_transform(const T* input,
                                T* output,
                                const dim_t size,
                                UnaryOp op) {
      if (size <= 0)
        return;
      const auto* device_input = device_cast(input);
      auto* device_output = device_cast(output);
      get_queue().parallel_for(::sycl::range<1>(static_cast<size_t>(size)),
                               [=](::sycl::id<1> id) {
        device_output[id] = op(device_input[id]);
      });
    }

    template <typename T, typename BinaryOp>
    inline void binary_transform(const T* a,
                                 const T* b,
                                 T* output,
                                 const dim_t size,
                                 BinaryOp op) {
      if (size <= 0)
        return;
      const auto* device_a = device_cast(a);
      const auto* device_b = device_cast(b);
      auto* device_output = device_cast(output);
      get_queue().parallel_for(::sycl::range<1>(static_cast<size_t>(size)),
                               [=](::sycl::id<1> id) {
        device_output[id] = op(device_a[id], device_b[id]);
      });
    }

  }
}
