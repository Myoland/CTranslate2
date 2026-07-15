#include "sycl/cache_gather_append.h"

#include <stdexcept>
#include <string>

#include "sycl/helpers.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace sycl_backend {
    namespace {

      void validate_cache_pair(const StorageView& keys,
                               const StorageView& values,
                               const char* name) {
        if (keys.shape() != values.shape() || keys.dtype() != values.dtype())
          throw std::invalid_argument(std::string(name)
                                      + " key and value caches must have the same shape and type");
      }

      template <typename T>
      void cache_gather_append_t(const StorageView& previous_keys,
                                 const StorageView& previous_values,
                                 const StorageView& indices,
                                 const StorageView& appended_keys,
                                 const StorageView& appended_values,
                                 const dim_t time_axis,
                                 StorageView& output_keys,
                                 StorageView& output_values) {
        using DT = device_type_t<T>;

        const DT* previous_k = device_cast(previous_keys.data<T>());
        const DT* previous_v = device_cast(previous_values.data<T>());
        const DT* appended_k = device_cast(appended_keys.data<T>());
        const DT* appended_v = device_cast(appended_values.data<T>());
        const int32_t* gather_indices = indices.data<int32_t>();
        DT* output_k = device_cast(output_keys.data<T>());
        DT* output_v = device_cast(output_values.data<T>());

        dim_t prefixes_per_batch = 1;
        for (dim_t i = 1; i < time_axis; ++i)
          prefixes_per_batch *= previous_keys.dim(i);

        const dim_t inner_size = previous_keys.stride(time_axis);
        const dim_t previous_span = previous_keys.dim(time_axis) * inner_size;
        const dim_t appended_span = appended_keys.dim(time_axis) * inner_size;
        const dim_t output_span = previous_span + appended_span;
        const dim_t output_size = output_keys.size();

        if (output_size == 0)
          return;

        get_queue().parallel_for(
          ::sycl::range<1>(static_cast<std::size_t>(output_size)),
          [=](::sycl::id<1> id) {
            const dim_t output_index = id[0];
            const dim_t output_prefix = output_index / output_span;
            const dim_t axis_inner = output_index % output_span;
            const dim_t batch = output_prefix / prefixes_per_batch;
            const dim_t prefix = output_prefix % prefixes_per_batch;

            if (axis_inner < previous_span) {
              const dim_t source_prefix
                = dim_t(gather_indices[batch]) * prefixes_per_batch + prefix;
              const dim_t source_index = source_prefix * previous_span + axis_inner;
              output_k[output_index] = previous_k[source_index];
              output_v[output_index] = previous_v[source_index];
            } else {
              const dim_t source_prefix = batch * prefixes_per_batch + prefix;
              const dim_t source_index
                = source_prefix * appended_span + axis_inner - previous_span;
              output_k[output_index] = appended_k[source_index];
              output_v[output_index] = appended_v[source_index];
            }
          });
      }

    }

    void cache_gather_append(const StorageView& previous_keys,
                             const StorageView& previous_values,
                             const StorageView& indices,
                             const StorageView& appended_keys,
                             const StorageView& appended_values,
                             dim_t time_axis,
                             StorageView& output_keys,
                             StorageView& output_values) {
      validate_cache_pair(previous_keys, previous_values, "Previous");
      validate_cache_pair(appended_keys, appended_values, "Appended");

      if (previous_keys.device() != Device::SYCL
          || appended_keys.device() != Device::SYCL
          || indices.device() != Device::SYCL)
        throw std::invalid_argument("cache_gather_append expects SYCL inputs");
      if (previous_keys.device_index() != appended_keys.device_index()
          || previous_keys.device_index() != indices.device_index())
        throw std::invalid_argument("cache_gather_append inputs must be on the same device");
      if (previous_keys.dtype() != appended_keys.dtype())
        throw std::invalid_argument("Previous and appended caches must have the same type");
      if (indices.dtype() != DataType::INT32 || indices.rank() != 1)
        throw std::invalid_argument("Cache gather indices must be a 1-D INT32 tensor");
      if (previous_keys.rank() != appended_keys.rank() || previous_keys.rank() < 2)
        throw std::invalid_argument("Previous and appended caches must have the same rank");

      if (time_axis < 0)
        time_axis += previous_keys.rank();
      if (time_axis <= 0 || time_axis >= previous_keys.rank())
        throw std::invalid_argument("Cache time axis must follow the batch axis");
      if (appended_keys.dim(0) != indices.size())
        throw std::invalid_argument("Appended cache batch size must match gather indices");

      for (dim_t i = 1; i < previous_keys.rank(); ++i) {
        if (i != time_axis && previous_keys.dim(i) != appended_keys.dim(i))
          throw std::invalid_argument("Previous and appended cache shapes are incompatible");
      }

      Shape output_shape(appended_keys.shape());
      output_shape[time_axis]
        = previous_keys.dim(time_axis) + appended_keys.dim(time_axis);
      output_keys.resize(output_shape);
      output_values.resize(std::move(output_shape));

      TYPE_DISPATCH(previous_keys.dtype(),
                    cache_gather_append_t<T>(previous_keys,
                                             previous_values,
                                             indices,
                                             appended_keys,
                                             appended_values,
                                             time_axis,
                                             output_keys,
                                             output_values));
    }

  }
}
