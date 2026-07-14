#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "ctranslate2/ops/activation.h"
#include "ctranslate2/types.h"
#include "sycl/helpers.h"

namespace ctranslate2 {
  namespace sycl_backend {

    template <typename T>
    inline T from_float(const float value) {
      return device_convert<T>(value);
    }

    inline size_t work_group_size(const size_t upper_bound = 256) {
      size_t size = std::min(upper_bound,
                             get_device().get_info<::sycl::info::device::max_work_group_size>());
      size_t power_of_two = 1;
      while ((power_of_two << 1) <= size)
        power_of_two <<= 1;
      return power_of_two;
    }

    inline float apply_activation(const float value, const ops::ActivationType type) {
      constexpr float sqrt_2_over_pi = 0.7978845608028654f;
      switch (type) {
      case ops::ActivationType::ReLU:
        return ::sycl::fmax(value, 0.f);
      case ops::ActivationType::GELUTanh: {
        const float inner = sqrt_2_over_pi * (value + 0.044715f * value * value * value);
        return 0.5f * value * (1.f + ::sycl::tanh(inner));
      }
      case ops::ActivationType::Swish:
        return value / (1.f + ::sycl::exp(-value));
      case ops::ActivationType::GELU:
        return 0.5f * value * (1.f + ::sycl::erf(value * 0.7071067811865475f));
      case ops::ActivationType::GELUSigmoid:
        return value / (1.f + ::sycl::exp(-1.702f * value));
      case ops::ActivationType::Tanh:
        return ::sycl::tanh(value);
      case ops::ActivationType::Sigmoid:
        return 1.f / (1.f + ::sycl::exp(-value));
      }
      return value;
    }

  }
}
