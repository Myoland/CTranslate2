#include "ctranslate2/ops/alibi_add.h"
#include "ctranslate2/ops/bias_add.h"
#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/median_filter.h"
#include "ctranslate2/ops/rotary.h"
#include "ctranslate2/ops/slide.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/tile.h"

#include <stdexcept>

#include "sycl/ops_utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void BiasAdd::compute(const StorageView& value,
                          const StorageView& bias,
                          StorageView& output,
                          const StorageView* residual) const {
      using DT = sycl_backend::device_type_t<T>;
      const dim_t axis = _axis < 0 ? value.rank() + _axis : _axis;
      dim_t width = 1;
      for (dim_t i = axis + 1; i < value.rank(); ++i)
        width *= value.dim(i);

      const DT* x = sycl_backend::device_cast(value.data<T>());
      const DT* b = sycl_backend::device_cast(bias.data<T>());
      const DT* r = residual ? sycl_backend::device_cast(residual->data<T>()) : nullptr;
      DT* y = sycl_backend::device_cast(output.data<T>());
      const dim_t depth = bias.size();
      const dim_t size = value.size();
      const bool activate = _activation_type != nullptr;
      const ActivationType activation = activate ? *_activation_type : ActivationType::ReLU;

      sycl_backend::get_queue().parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> id) {
        const dim_t i = id[0];
        float v = sycl_backend::to_float(x[i])
          + sycl_backend::to_float(b[(i / width) % depth]);
        if (r)
          v += sycl_backend::to_float(r[i]);
        if (activate)
          v = sycl_backend::apply_activation(v, activation);
        y[i] = sycl_backend::from_float<DT>(v);
      });
    }

    template <Device D, typename T>
    void AlibiAdd::compute(const StorageView& input,
                           const StorageView& alibi,
                           const dim_t alibi_offset,
                           StorageView& output) const {
      using DT = sycl_backend::device_type_t<T>;
      const DT* x = sycl_backend::device_cast(input.data<T>());
      const DT* a = sycl_backend::device_cast(alibi.data<T>());
      DT* y = sycl_backend::device_cast(output.data<T>());
      const dim_t num_heads = input.dim(1);
      const dim_t query_length = input.dim(2);
      const dim_t key_length = input.dim(3);
      const dim_t cached_key_length = alibi.dim(-1);
      const dim_t size = input.size();

      sycl_backend::get_queue().parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> id) {
        const dim_t index = id[0];
        const dim_t row = index / key_length;
        const dim_t head = (row / query_length) % num_heads;
        const dim_t col = index % key_length;
        y[index] = sycl_backend::from_float<DT>(
          sycl_backend::to_float(x[index])
          + sycl_backend::to_float(a[head * cached_key_length + alibi_offset + col]));
      });
    }

    template <typename DT>
    static void copy_axis_slice(const DT* input,
                                DT* output,
                                const dim_t input_dim,
                                const dim_t output_dim,
                                const dim_t inner_size,
                                const dim_t offset,
                                const dim_t output_size,
                                const bool scatter) {
      sycl_backend::get_queue().parallel_for(::sycl::range<1>(output_size), [=](::sycl::id<1> id) {
        const dim_t i = id[0];
        if (scatter) {
          const dim_t outer = i / (input_dim * inner_size);
          const dim_t axis = (i / inner_size) % input_dim;
          const dim_t inner = i % inner_size;
          output[(outer * output_dim + axis + offset) * inner_size + inner] = input[i];
        } else {
          const dim_t outer = i / (output_dim * inner_size);
          const dim_t axis = (i / inner_size) % output_dim;
          const dim_t inner = i % inner_size;
          output[i] = input[(outer * input_dim + axis + offset) * inner_size + inner];
        }
      });
    }

    template <Device D, typename T>
    void Concat::compute(const std::vector<const StorageView*>& inputs,
                         StorageView& output) const {
      using DT = sycl_backend::device_type_t<T>;
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      const dim_t output_dim = output.dim(axis);
      const dim_t inner_size = output.stride(axis);
      DT* output_data = sycl_backend::device_cast(output.data<T>());
      dim_t offset = 0;
      for (const StorageView* input : inputs) {
        const DT* input_data = sycl_backend::device_cast(input->data<T>());
        if (axis == 0) {
          primitives<D>::copy(input->data<T>(), output.data<T>() + offset, input->size());
          offset += input->size();
        } else {
          const dim_t input_dim = input->dim(axis);
          copy_axis_slice(input_data, output_data, input_dim, output_dim,
                          inner_size, offset, input->size(), true);
          offset += input_dim;
        }
      }
    }

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      using DT = sycl_backend::device_type_t<T>;
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t input_dim = input.dim(axis);
      const dim_t inner_size = input.stride(axis);
      const DT* input_data = sycl_backend::device_cast(input.data<T>());
      dim_t offset = 0;
      for (StorageView* output : outputs) {
        if (axis == 0) {
          primitives<D>::copy(input.data<T>() + offset, output->data<T>(), output->size());
          offset += output->size();
        } else {
          const dim_t output_dim = output->dim(axis);
          copy_axis_slice(input_data, sycl_backend::device_cast(output->data<T>()),
                          input_dim, output_dim, inner_size, offset, output->size(), false);
          offset += output_dim;
        }
      }
    }

    template <Device D, typename T>
    void Slide::compute(const StorageView& input,
                        StorageView& output,
                        const dim_t& index) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t inner_size = input.stride(axis) == 0 ? 1 : input.stride(axis);
      if (axis == 0) {
        primitives<D>::copy(input.data<T>() + index * input.stride(axis),
                            output.data<T>(), output.size());
      } else {
        copy_axis_slice(sycl_backend::device_cast(input.data<T>()),
                        sycl_backend::device_cast(output.data<T>()),
                        input.dim(axis), output.dim(axis), inner_size,
                        index, output.size(), false);
      }
    }

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      if (axis != batch_dims)
        throw std::invalid_argument("Gather only supports indexing the first non batch dimension");

      using DT = sycl_backend::device_type_t<T>;
      const DT* src = sycl_backend::device_cast(data.data<T>());
      const int32_t* indices = input.data<int32_t>();
      DT* dst = sycl_backend::device_cast(output.data<T>());
      const dim_t batch_stride = axis > 0 ? data.stride(axis - 1) : data.size();
      const dim_t batch_size = data.size() / batch_stride;
      const dim_t num_indices_per_batch = input.size() / batch_size;
      const dim_t gather_size = data.stride(axis);
      const dim_t size = output.size();

      sycl_backend::get_queue().parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> id) {
        const dim_t i = id[0];
        const dim_t inner = i % gather_size;
        const dim_t outer = i / gather_size;
        const dim_t batch = outer / num_indices_per_batch;
        dst[i] = src[batch * batch_stride + dim_t(indices[outer]) * gather_size + inner];
      });
    }

    template <Device D, typename T>
    void Tile::compute(const StorageView& input,
                       const dim_t,
                       const dim_t inner_size,
                       StorageView& output) const {
      using DT = sycl_backend::device_type_t<T>;
      const DT* src = sycl_backend::device_cast(input.data<T>());
      DT* dst = sycl_backend::device_cast(output.data<T>());
      const dim_t num_tiles = _num_tiles;
      sycl_backend::get_queue().parallel_for(::sycl::range<1>(output.size()), [=](::sycl::id<1> id) {
        const dim_t i = id[0];
        const dim_t row = i / (inner_size * num_tiles);
        dst[i] = src[row * inner_size + i % inner_size];
      });
    }

    template <Device D, typename T>
    void Rotary::compute(const StorageView& input,
                         const StorageView& sin,
                         const StorageView& cos,
                         StorageView& output,
                         bool is_transposed) const {
      using DT = sycl_backend::device_type_t<T>;
      const DT* x = sycl_backend::device_cast(input.data<T>());
      const DT* s = sycl_backend::device_cast(sin.data<T>());
      const DT* c = sycl_backend::device_cast(cos.data<T>());
      DT* y = sycl_backend::device_cast(output.data<T>());
      const dim_t max_time = is_transposed ? input.dim(-2) : input.dim(-3);
      const dim_t head_size = is_transposed ? input.dim(-3) : input.dim(-2);
      const dim_t depth = input.dim(-1);
      const dim_t ndims = _ndims == 0 ? depth : _ndims;
      const dim_t middle = ndims / 2;
      const bool interleave = _interleave;

      sycl_backend::get_queue().parallel_for(::sycl::range<1>(input.size()), [=](::sycl::id<1> id) {
        const dim_t index = id[0];
        const dim_t row = index / depth;
        const dim_t col = index % depth;
        if (col >= ndims) {
          y[index] = x[index];
          return;
        }
        const dim_t time = is_transposed ? row % max_time : row / head_size;
        const dim_t paired = interleave
          ? (col % 2 == 0 ? col + 1 : col - 1)
          : (col < middle ? col + middle : col - middle);
        const float rotated = (interleave ? (col % 2 == 0) : (col < middle))
          ? -sycl_backend::to_float(x[row * depth + paired])
          : sycl_backend::to_float(x[row * depth + paired]);
        const float value = sycl_backend::to_float(x[index]) * sycl_backend::to_float(c[time * ndims + col])
          + rotated * sycl_backend::to_float(s[time * ndims + col]);
        y[index] = sycl_backend::from_float<DT>(value);
      });
    }

    template <Device D, typename T>
    void MedianFilter::compute(const StorageView& input,
                               const dim_t axis_size,
                               StorageView& output) const {
      constexpr int max_window = 129;
      const int depth = static_cast<int>(axis_size);
      const int width = static_cast<int>(_width);
      const int rank = width / 2;
      if (width <= 1 || depth <= rank) {
        if (&output != &input)
          output.copy_from(input);
        return;
      }
      if ((width & 1) == 0)
        throw std::invalid_argument("MedianFilter width must be odd");
      if (width > max_window)
        throw std::invalid_argument("MedianFilter width exceeds supported SYCL max (129)");

      using DT = sycl_backend::device_type_t<T>;
      const DT* x = sycl_backend::device_cast(input.data<T>());
      DT* y = sycl_backend::device_cast(output.data<T>());
      const dim_t size = input.size();
      sycl_backend::get_queue().parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> id) {
        const dim_t index = id[0];
        const int col = index % depth;
        const dim_t row_offset = index - col;
        float window[max_window];
        for (int k = -rank; k <= rank; ++k) {
          int read = col + k;
          if (read < 0)
            read = -read;
          if (read >= depth)
            read = 2 * depth - read - 2;
          window[k + rank] = sycl_backend::to_float(x[row_offset + read]);
        }
        for (int i = 1; i < width; ++i) {
          const float key = window[i];
          int j = i - 1;
          while (j >= 0 && window[j] > key) {
            window[j + 1] = window[j];
            --j;
          }
          window[j + 1] = key;
        }
        y[index] = sycl_backend::from_float<DT>(window[rank]);
      });
    }

#define DECLARE_FLOAT_OPS(T)                                           \
    template void BiasAdd::compute<Device::SYCL, T>(                   \
      const StorageView&, const StorageView&, StorageView&, const StorageView*) const; \
    template void AlibiAdd::compute<Device::SYCL, T>(                  \
      const StorageView&, const StorageView&, dim_t, StorageView&) const; \
    template void Rotary::compute<Device::SYCL, T>(                    \
      const StorageView&, const StorageView&, const StorageView&, StorageView&, bool) const; \
    template void MedianFilter::compute<Device::SYCL, T>(              \
      const StorageView&, dim_t, StorageView&) const;

    DECLARE_FLOAT_OPS(float)
    DECLARE_FLOAT_OPS(float16_t)
    DECLARE_FLOAT_OPS(bfloat16_t)

#define DECLARE_TYPED_OPS(T)                                           \
    template void Concat::compute<Device::SYCL, T>(                    \
      const std::vector<const StorageView*>&, StorageView&) const;     \
    template void Split::compute<Device::SYCL, T>(                     \
      const StorageView&, std::vector<StorageView*>&) const;           \
    template void Slide::compute<Device::SYCL, T>(                     \
      const StorageView&, StorageView&, const dim_t&) const;           \
    template void Gather::compute<Device::SYCL, T>(                    \
      const StorageView&, const StorageView&, dim_t, dim_t, StorageView&) const; \
    template void Tile::compute<Device::SYCL, T>(                      \
      const StorageView&, dim_t, dim_t, StorageView&) const;

    DECLARE_ALL_TYPES(DECLARE_TYPED_OPS)

#undef DECLARE_FLOAT_OPS
#undef DECLARE_TYPED_OPS

  }
}
