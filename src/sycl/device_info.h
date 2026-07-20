#pragma once

#include <cstdint>

namespace ctranslate2 {
  namespace sycl_backend {

    enum class IntelGpuModel {
      Other,
      ArcB580,
      ArcB390,
      Arc140V,
    };

    // Classifies the exact Intel GPU product used by tuned kernels. Keeping
    // this mapping separate also makes the dispatch policy testable without a
    // physical GPU.
    IntelGpuModel intel_gpu_model_from_device_id(
      uint32_t device_id,
      uint32_t execution_units = 0);

    // Returns the cached model classification for a selected runtime device.
    IntelGpuModel get_intel_gpu_model(int device_index = -1);

  }
}
