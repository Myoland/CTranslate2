#pragma once

#include "ctranslate2/storage_view.h"
#include "sycl/device_info.h"

namespace ctranslate2 {
  namespace sycl_backend {

    struct FusedQKVProjectionFP16Config {
      dim_t rows;
      dim_t input_size;
      dim_t output_size;
    };

    // Returns whether the specialized projection is profitable for a product
    // and row count. This helper is independent of the selected device so the
    // measured dispatch boundaries can be tested without GPU hardware.
    bool is_fused_qkv_projection_profitable(IntelGpuModel model, dim_t rows);

    // This predicate is intentionally narrow: the kernel relies on a 16x16
    // FP16 matrix layout and uses independently measured B580, B390, and
    // Arc 140V row boundaries for Whisper's iterative self-attention projection.
    bool supports_fused_qkv_projection_fp16(
      const FusedQKVProjectionFP16Config& config,
      int device_index = -1);

    // Computes input * weight^T + bias and writes the 3 equal output parts
    // directly, without materializing the combined projection. Returns false
    // without submitting work when the exact shape or device is unsupported.
    bool fused_qkv_projection_fp16(const StorageView& input,
                                   const StorageView& weight,
                                   const StorageView& bias,
                                   StorageView& output1,
                                   StorageView& output2,
                                   StorageView& output3);

  }
}
