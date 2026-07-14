#include "ctranslate2/ops/bias_add.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {
    namespace {

      bool storage_ranges_overlap(const StorageView& a, const StorageView& b) {
        if (&a == &b)
          return true;
        if (!a.buffer() || !b.buffer())
          return false;

        const auto a_begin = reinterpret_cast<std::uintptr_t>(a.buffer());
        const auto b_begin = reinterpret_cast<std::uintptr_t>(b.buffer());
        const auto a_bytes = static_cast<std::uintptr_t>(a.reserved_memory());
        const auto b_bytes = static_cast<std::uintptr_t>(b.reserved_memory());
        if (a_bytes > std::numeric_limits<std::uintptr_t>::max() - a_begin
            || b_bytes > std::numeric_limits<std::uintptr_t>::max() - b_begin)
          return true;
        const auto a_end = a_begin + a_bytes;
        const auto b_end = b_begin + b_bytes;
        return a_begin < b_end && b_begin < a_end;
      }

    }

    BiasAdd::BiasAdd(const ActivationType* activation_type, const dim_t axis)
      : _activation_type(activation_type)
      , _axis(axis)
    {
    }

    void BiasAdd::operator()(const StorageView& value,
                             const StorageView& bias,
                             StorageView& output,
                             const StorageView* residual) const {
      PROFILE("BiasAdd");
      output.resize_as(value);

      DEVICE_AND_FLOAT_DISPATCH("BiasAdd", value.device(), value.dtype(),
                                (compute<D, T>(value, bias, output, residual)));
    }

    void BiasAdd::split3(const StorageView& value,
                         const StorageView& bias,
                         StorageView& output1,
                         StorageView& output2,
                         StorageView& output3) const {
#ifdef CT2_WITH_SYCL
      PROFILE("BiasAddSplit3");

      if (_activation_type)
        throw std::invalid_argument("BiasAdd::split3 does not support an activation");
      if (value.device() != Device::SYCL || value.dtype() != DataType::FLOAT16)
        throw std::invalid_argument("BiasAdd::split3 only supports SYCL FLOAT16 values");
      if (bias.device() != value.device()
          || bias.device_index() != value.device_index()
          || bias.dtype() != value.dtype())
        throw std::invalid_argument("BiasAdd::split3 requires the bias to match the value type and device");
      if (value.rank() == 0)
        throw std::invalid_argument("BiasAdd::split3 requires a non-scalar value");

      const dim_t axis = _axis < 0 ? value.rank() + _axis : _axis;
      if (axis != value.rank() - 1)
        throw std::invalid_argument("BiasAdd::split3 only supports the last dimension");

      const dim_t depth = value.dim(axis);
      if (bias.rank() != 1 || bias.size() != depth)
        throw std::invalid_argument("BiasAdd::split3 requires a 1D bias matching the last dimension");
      if (depth % 3 != 0)
        throw std::invalid_argument("BiasAdd::split3 requires 3 equal output parts");

      for (const StorageView* output : {&output1, &output2, &output3}) {
        if (output->device() != value.device()
            || output->device_index() != value.device_index()
            || output->dtype() != value.dtype())
          throw std::invalid_argument(
            "BiasAdd::split3 requires outputs matching the value type and device");
        if (storage_ranges_overlap(value, *output)
            || storage_ranges_overlap(bias, *output))
          throw std::invalid_argument("BiasAdd::split3 does not support aliased outputs");
      }
      if (storage_ranges_overlap(output1, output2)
          || storage_ranges_overlap(output1, output3)
          || storage_ranges_overlap(output2, output3))
        throw std::invalid_argument("BiasAdd::split3 requires distinct outputs");

      Shape output_shape = value.shape();
      output_shape.back() = depth / 3;
      output1.resize(output_shape);
      output2.resize(output_shape);
      output3.resize(std::move(output_shape));

      compute_split3<float16_t>(value, bias, output1, output2, output3);
#else
      (void)value;
      (void)bias;
      (void)output1;
      (void)output2;
      (void)output3;
      throw std::runtime_error("BiasAdd::split3 requires a SYCL build");
#endif
    }

  }
}
