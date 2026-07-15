#include "sycl/fused_qkv_projection.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include "sycl/helpers.h"

namespace ctranslate2 {
  namespace sycl_backend {
    namespace {

      namespace matrix = ::sycl::ext::oneapi::experimental::matrix;
      namespace intel_matrix = ::sycl::ext::intel::experimental::matrix;

      constexpr dim_t projection_rows = 5;
      constexpr dim_t projection_input_size = 1280;
      constexpr dim_t projection_output_size = 3840;
      constexpr dim_t projection_part_size = projection_output_size / 3;
      constexpr size_t matrix_tile_size = 16;
      constexpr size_t matrix_subgroup_size = 16;
      constexpr size_t required_pointer_alignment = 16;

      class FusedQKVProjectionFP16Kernel;

      bool device_supports_kernel(const int device_index) {
        struct Cache {
          int device_index = -2;
          bool supported = false;
        };
        thread_local Cache cache;
        if (cache.device_index == device_index)
          return cache.supported;

        const ::sycl::device& device = get_device(device_index);
        const std::string name = device.get_info<::sycl::info::device::name>();
        bool supported = name.find("B580") != std::string::npos
                         && device.has(::sycl::aspect::fp16)
                         && device.has(::sycl::aspect::ext_intel_matrix);
        if (supported) {
          const auto combinations = device.get_info<
            ::sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
          supported = std::any_of(
            combinations.begin(), combinations.end(), [](const auto& combination) {
              return combination.msize == matrix_tile_size
                     && combination.nsize == matrix_tile_size
                     && combination.ksize == matrix_tile_size
                     && combination.atype == matrix::matrix_type::fp16
                     && combination.btype == matrix::matrix_type::fp16
                     && combination.ctype == matrix::matrix_type::fp16
                     && combination.dtype == matrix::matrix_type::fp16;
            });
        }

        cache.device_index = device_index;
        cache.supported = supported;
        return supported;
      }

      bool is_aligned(const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer) % required_pointer_alignment == 0;
      }

    }

    bool supports_fused_qkv_projection_fp16(
      const FusedQKVProjectionFP16Config& config,
      int device_index) {
      if (config.rows != projection_rows
          || config.input_size != projection_input_size
          || config.output_size != projection_output_size)
        return false;
      if (device_index < 0)
        device_index = get_device_index();
      return device_supports_kernel(device_index);
    }

    bool fused_qkv_projection_fp16(const StorageView& input,
                                   const StorageView& weight,
                                   const StorageView& bias,
                                   StorageView& output1,
                                   StorageView& output2,
                                   StorageView& output3) {
      if (input.device() != Device::SYCL
          || input.dtype() != DataType::FLOAT16
          || input.rank() != 3
          || input.dim(0) != projection_rows
          || input.dim(1) != 1
          || input.dim(2) != projection_input_size
          || weight.device() != input.device()
          || weight.device_index() != input.device_index()
          || weight.dtype() != input.dtype()
          || weight.shape() != Shape({projection_output_size,
                                      projection_input_size})
          || bias.device() != input.device()
          || bias.device_index() != input.device_index()
          || bias.dtype() != input.dtype()
          || bias.shape() != Shape({projection_output_size})
          || output1.device() != input.device()
          || output2.device() != input.device()
          || output3.device() != input.device()
          || output1.device_index() != input.device_index()
          || output2.device_index() != input.device_index()
          || output3.device_index() != input.device_index()
          || output1.dtype() != input.dtype()
          || output2.dtype() != input.dtype()
          || output3.dtype() != input.dtype())
        return false;

      const FusedQKVProjectionFP16Config config{
        projection_rows, projection_input_size, projection_output_size};
      if (!supports_fused_qkv_projection_fp16(config, input.device_index()))
        return false;

      const auto* device_input = device_cast(input.data<float16_t>());
      const auto* device_weight = device_cast(weight.data<float16_t>());
      const auto* device_bias = device_cast(bias.data<float16_t>());
      if (!is_aligned(device_input) || !is_aligned(device_weight))
        return false;

      Shape output_shape(input.shape());
      output_shape.back() = projection_part_size;
      output1.resize(output_shape);
      output2.resize(output_shape);
      output3.resize(std::move(output_shape));

      auto* device_output1 = device_cast(output1.data<float16_t>());
      auto* device_output2 = device_cast(output2.data<float16_t>());
      auto* device_output3 = device_cast(output3.data<float16_t>());
      if (!is_aligned(device_output1)
          || !is_aligned(device_output2)
          || !is_aligned(device_output3))
        return false;

      get_queue(input.device_index()).submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<::sycl::half, 1> tile_storage(
          matrix_tile_size * matrix_tile_size, handler);
        handler.parallel_for<FusedQKVProjectionFP16Kernel>(
          ::sycl::nd_range<1>(
            ::sycl::range<1>((projection_output_size / matrix_tile_size)
                             * matrix_subgroup_size),
            ::sycl::range<1>(matrix_subgroup_size)),
          [=](::sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(matrix_subgroup_size)]] {
          const auto subgroup = item.get_sub_group();
          const size_t column_tile = item.get_group(0);
          const size_t global_column = column_tile * matrix_tile_size;

          const auto global_input = ::sycl::address_space_cast<
            ::sycl::access::address_space::global_space,
            ::sycl::access::decorated::no>(device_input);
          const auto global_weight = ::sycl::address_space_cast<
            ::sycl::access::address_space::global_space,
            ::sycl::access::decorated::no>(device_weight);

          matrix::joint_matrix<::sycl::sub_group,
                               ::sycl::half,
                               matrix::use::a,
                               matrix_tile_size,
                               matrix_tile_size,
                               matrix::layout::row_major> tile_a;
          matrix::joint_matrix<::sycl::sub_group,
                               ::sycl::half,
                               matrix::use::b,
                               matrix_tile_size,
                               matrix_tile_size,
                               matrix::layout::col_major> tile_b;
          matrix::joint_matrix<::sycl::sub_group,
                               ::sycl::half,
                               matrix::use::accumulator,
                               matrix_tile_size,
                               matrix_tile_size> tile_c;
          matrix::joint_matrix_fill(subgroup, tile_c, ::sycl::half(0));
          for (size_t k = 0; k < projection_input_size; k += matrix_tile_size) {
            // The allocation contains only 5 rows. Checked load zero-fills the
            // other 11 tile rows without reading beyond input.size().
            intel_matrix::joint_matrix_load_checked(subgroup,
                                                     tile_a,
                                                     global_input,
                                                     projection_input_size,
                                                     projection_rows,
                                                     projection_input_size,
                                                     0,
                                                     k);
            matrix::joint_matrix_load(
              subgroup,
              tile_b,
              global_weight + global_column * projection_input_size + k,
              projection_input_size);
            matrix::joint_matrix_mad(subgroup, tile_c, tile_a, tile_b, tile_c);
          }

          const size_t part = global_column / projection_part_size;
          const size_t part_column = global_column % projection_part_size;
          ::sycl::half* output = part == 0
                                  ? device_output1
                                  : (part == 1 ? device_output2 : device_output3);
          const auto local_output = tile_storage.template get_multi_ptr<
            ::sycl::access::decorated::no>();
          matrix::joint_matrix_store(subgroup,
                                     tile_c,
                                     local_output,
                                     matrix_tile_size,
                                     matrix::layout::row_major);
          item.barrier(::sycl::access::fence_space::local_space);

          const size_t local_column = item.get_local_id(0);
          const ::sycl::half column_bias
            = device_bias[global_column + local_column];
          for (size_t row = 0; row < projection_rows; ++row) {
            const ::sycl::half element
              = tile_storage[row * matrix_tile_size + local_column];
            output[row * projection_part_size + part_column + local_column]
              = ::sycl::half(static_cast<float>(element)
                             + static_cast<float>(column_bias));
          }
        });
      });
      return true;
    }

  }
}
