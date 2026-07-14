#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/gemm.h"

#include "sycl/ops_utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Conv1D::compute(const StorageView& input,
                         const StorageView& weight,
                         const StorageView* bias,
                         StorageView& output,
                         const StorageView* qscale) const {
      if (qscale)
        throw std::runtime_error("Quantized Conv1D weights are not supported by the SYCL backend");

      using DT = sycl_backend::device_type_t<T>;
      const dim_t batch_size = input.dim(0);
      const dim_t in_channels = input.dim(1);
      const dim_t input_length = input.dim(2);
      const dim_t out_channels = weight.dim(0);
      const dim_t kernel_size = weight.dim(2);
      const dim_t output_length = output.dim(2);
      const dim_t in_channels_per_group = in_channels / _groups;
      const dim_t out_channels_per_group = out_channels / _groups;
      const dim_t k = in_channels_per_group * kernel_size;

      StorageView columns({batch_size, _groups, output_length, k},
                          DataTypeToEnum<T>::value,
                          D);
      const DT* x = sycl_backend::device_cast(input.data<T>());
      DT* p = sycl_backend::device_cast(columns.data<T>());
      const dim_t groups = _groups;
      const dim_t stride = _stride;
      const dim_t padding = _padding;
      const dim_t dilation = _dilation;
      const dim_t size = columns.size();

      sycl_backend::get_queue().parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> id) {
        const dim_t index = id[0];
        const dim_t kk = index % k;
        const dim_t output_position = (index / k) % output_length;
        const dim_t group = (index / (k * output_length)) % groups;
        const dim_t batch = index / (k * output_length * groups);
        const dim_t channel = kk / kernel_size;
        const dim_t kernel_position = kk % kernel_size;
        const dim_t input_position = output_position * stride - padding + kernel_position * dilation;
        if (input_position < 0 || input_position >= input_length) {
          p[index] = sycl_backend::from_float<DT>(0.f);
        } else {
          const dim_t input_channel = group * in_channels_per_group + channel;
          p[index] = x[(batch * in_channels + input_channel) * input_length + input_position];
        }
      });

      const T* w = weight.data<T>();
      const T* column_data = columns.data<T>();
      T* result = output.data<T>();
      const dim_t weight_group_stride = out_channels_per_group * k;
      const dim_t column_group_stride = k * output_length;
      const dim_t output_group_stride = out_channels_per_group * output_length;
      for (dim_t group = 0; group < groups; ++group) {
        primitives<D>::gemm_batch_strided(
          false,
          true,
          out_channels_per_group,
          output_length,
          k,
          1.f,
          w + group * weight_group_stride,
          k,
          0,
          column_data + group * column_group_stride,
          k,
          groups * column_group_stride,
          0.f,
          result + group * output_group_stride,
          output_length,
          groups * output_group_stride,
          batch_size);
      }

      apply_bias_and_activation(output, bias, _activation_type, nullptr, -2);
    }

#define DECLARE_IMPL(T)                                                 \
    template void Conv1D::compute<Device::SYCL, T>(                    \
      const StorageView&, const StorageView&, const StorageView*,      \
      StorageView&, const StorageView*) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

  }
}
