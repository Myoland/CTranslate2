#include "sycl/fused_qkv_projection.h"

#include <algorithm>
#include <cstdint>

#include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include "sycl/helpers.h"

namespace ctranslate2 {
  namespace sycl_backend {
    namespace {

      namespace matrix = ::sycl::ext::oneapi::experimental::matrix;
      namespace intel_matrix = ::sycl::ext::intel::experimental::matrix;

      // faster-whisper expands each audio chunk by the beam size before the
      // iterative decoder starts.  With its usual beam size of 5, a batched
      // request therefore reaches this projection with 5 * batch_size rows.
      // Batches are also compacted as individual chunks finish, so accept all
      // row counts through a product-specific profitability boundary instead
      // of a few exact multiples of 5.
      constexpr dim_t maximum_projection_rows = 80;
      constexpr dim_t projection_input_size = 1280;
      constexpr dim_t projection_output_size = 3840;
      constexpr dim_t projection_part_size = projection_output_size / 3;
      constexpr size_t matrix_tile_size = 16;
      constexpr size_t matrix_subgroup_size = 16;
      constexpr size_t required_pointer_alignment = 16;

      template <typename Accumulator>
      class FusedQKVProjectionKernel;

      bool uses_fp32_accumulator(const IntelGpuModel model) {
        return model == IntelGpuModel::ArcB390
               || model == IntelGpuModel::Arc140V;
      }

      bool has_fused_qkv_projection_policy(const IntelGpuModel model) {
        return model == IntelGpuModel::ArcB580
               || model == IntelGpuModel::ArcB390
               || model == IntelGpuModel::Arc140V;
      }

      bool device_supports_kernel(const int device_index) {
        struct Cache {
          int device_index = -2;
          bool supported = false;
        };
        thread_local Cache cache;
        if (cache.device_index == device_index)
          return cache.supported;

        const ::sycl::device& device = get_device(device_index);
        const IntelGpuModel model = get_intel_gpu_model(device_index);
        const auto subgroup_sizes
          = device.get_info<::sycl::info::device::sub_group_sizes>();
        const matrix::matrix_type accumulator_type
          = uses_fp32_accumulator(model)
              ? matrix::matrix_type::fp32
              : matrix::matrix_type::fp16;
        const size_t accumulator_size
          = uses_fp32_accumulator(model)
              ? sizeof(float)
              : sizeof(::sycl::half);
        bool supported = has_fused_qkv_projection_policy(model)
                         && device.has(::sycl::aspect::fp16)
                         && device.has(::sycl::aspect::ext_intel_matrix)
                         && device.get_info<
                              ::sycl::info::device::max_work_group_size>()
                              >= matrix_subgroup_size
                         && device.get_info<
                              ::sycl::info::device::max_work_item_sizes<1>>()[0]
                              >= matrix_subgroup_size
                         && std::find(subgroup_sizes.begin(),
                                      subgroup_sizes.end(),
                                      matrix_subgroup_size)
                              != subgroup_sizes.end()
                         && device.get_info<
                              ::sycl::info::device::local_mem_size>()
                              >= matrix_tile_size * matrix_tile_size
                                   * accumulator_size;
        if (supported) {
          const auto combinations = device.get_info<
            ::sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
          supported = std::any_of(
            combinations.begin(), combinations.end(), [&](const auto& combination) {
              return combination.msize == matrix_tile_size
                     && combination.nsize == matrix_tile_size
                     && combination.ksize == matrix_tile_size
                     && combination.atype == matrix::matrix_type::fp16
                     && combination.btype == matrix::matrix_type::fp16
                     && combination.ctype == accumulator_type
                     && combination.dtype == accumulator_type;
            });
        }

        cache.device_index = device_index;
        cache.supported = supported;
        return supported;
      }

      bool is_aligned(const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer) % required_pointer_alignment == 0;
      }

      template <typename Accumulator>
      void launch_fused_qkv_projection(const int device_index,
                                       const dim_t projection_rows,
                                       const ::sycl::half* device_input,
                                       const ::sycl::half* device_weight,
                                       const ::sycl::half* device_bias,
                                       ::sycl::half* device_output1,
                                       ::sycl::half* device_output2,
                                       ::sycl::half* device_output3) {
        const size_t row_tiles
          = (static_cast<size_t>(projection_rows) + matrix_tile_size - 1)
            / matrix_tile_size;
        const size_t column_tiles = projection_output_size / matrix_tile_size;
        get_queue(device_index).submit([&](::sycl::handler& handler) {
          ::sycl::local_accessor<Accumulator, 1> tile_storage(
            matrix_tile_size * matrix_tile_size, handler);
          handler.parallel_for<FusedQKVProjectionKernel<Accumulator>>(
            ::sycl::nd_range<1>(
              ::sycl::range<1>(row_tiles * column_tiles
                               * matrix_subgroup_size),
              ::sycl::range<1>(matrix_subgroup_size)),
            [=](::sycl::nd_item<1> item)
              [[sycl::reqd_sub_group_size(matrix_subgroup_size)]] {
            const auto subgroup = item.get_sub_group();
            const size_t row_tile = item.get_group(0) / column_tiles;
            const size_t column_tile = item.get_group(0) % column_tiles;
            const size_t global_row = row_tile * matrix_tile_size;
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
                                 Accumulator,
                                 matrix::use::accumulator,
                                 matrix_tile_size,
                                 matrix_tile_size> tile_c;
            matrix::joint_matrix_fill(subgroup, tile_c, Accumulator(0));
            for (size_t k = 0; k < projection_input_size;
                 k += matrix_tile_size) {
              // The final row tile can be partial. Checked load zero-fills its
              // remaining rows without reading beyond input.size().
              intel_matrix::joint_matrix_load_checked(subgroup,
                                                       tile_a,
                                                       global_input,
                                                       projection_input_size,
                                                       projection_rows,
                                                       projection_input_size,
                                                       global_row,
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
            const size_t rows_in_tile
              = ::sycl::min(matrix_tile_size,
                            static_cast<size_t>(projection_rows) - global_row);
            for (size_t row = 0; row < rows_in_tile; ++row) {
              const Accumulator element
                = tile_storage[row * matrix_tile_size + local_column];
              output[(global_row + row) * projection_part_size
                     + part_column + local_column]
                = ::sycl::half(static_cast<float>(element)
                               + static_cast<float>(column_bias));
            }
          });
        });
      }

    }

    bool is_fused_qkv_projection_profitable(const IntelGpuModel model,
                                             const dim_t rows) {
      switch (model) {
      case IntelGpuModel::ArcB580:
        // The crossover with oneMKL is between 80 and 96 rows on B580.
        return rows >= 1 && rows <= 80;
      case IntelGpuModel::ArcB390:
        // Xe3 wins only for the small decoder batches represented by 5-20
        // rows. oneMKL is faster for one row and from 21 rows onward.
        return rows >= 5 && rows <= 20;
      case IntelGpuModel::Arc140V:
        // On Lunar Lake, the FP32-accumulator kernel wins through the common
        // beam-5 batch of 20 rows. oneMKL overtakes it from 21 rows onward.
        return rows >= 1 && rows <= 20;
      default:
        return false;
      }
    }

    bool supports_fused_qkv_projection_fp16(
      const FusedQKVProjectionFP16Config& config,
      int device_index) {
      if (config.rows <= 0
          || config.rows > maximum_projection_rows
          || config.input_size != projection_input_size
          || config.output_size != projection_output_size)
        return false;
      if (device_index < 0)
        device_index = get_device_index();
      if (!is_fused_qkv_projection_profitable(
            get_intel_gpu_model(device_index), config.rows))
        return false;
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
          || input.dim(0) <= 0
          || input.dim(0) > maximum_projection_rows
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

      const dim_t projection_rows = input.dim(0);
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

      // B390 and Arc 140V both run faster and closer to oneMKL with an FP32
      // matrix accumulator. Keep B580 on its established FP16 path.
      if (uses_fp32_accumulator(get_intel_gpu_model(input.device_index()))) {
        launch_fused_qkv_projection<float>(input.device_index(),
                                           projection_rows,
                                           device_input,
                                           device_weight,
                                           device_bias,
                                           device_output1,
                                           device_output2,
                                           device_output3);
      } else {
        launch_fused_qkv_projection<::sycl::half>(input.device_index(),
                                                  projection_rows,
                                                  device_input,
                                                  device_weight,
                                                  device_bias,
                                                  device_output1,
                                                  device_output2,
                                                  device_output3);
      }
      return true;
    }

  }
}
