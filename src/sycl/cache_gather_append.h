#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace sycl_backend {

    // Gathers the batch dimension of a pair of caches and appends new values
    // along time_axis. Both output caches are produced by one kernel launch.
    void cache_gather_append(const StorageView& previous_keys,
                             const StorageView& previous_values,
                             const StorageView& indices,
                             const StorageView& appended_keys,
                             const StorageView& appended_values,
                             dim_t time_axis,
                             StorageView& output_keys,
                             StorageView& output_values);

  }
}
