#pragma once

#include <cstddef>
#include <string>

#include <sycl/sycl.hpp>

#include "sycl/device_info.h"

namespace ctranslate2 {
  namespace sycl_backend {

    // Only Intel Level Zero GPU devices with device USM are exposed.
    int get_device_count();
    bool has_gpu();

    int get_device_index();
    void set_device_index(int index);

    const ::sycl::device& get_device(int device_index = -1);
    const ::sycl::context& get_context(int device_index = -1);

    // Returns the in-order queue owned by this host thread for the selected
    // device. The context and device objects are process-wide and stable.
    ::sycl::queue& get_queue(int device_index = -1);

    // Waits all registered queues for a device or just the current thread's
    // queue, respectively. Both functions surface asynchronous errors.
    void synchronize(int device_index = -1);
    void synchronize_queue();
    void check_async_errors();

    // Flushes outstanding work. Registry objects intentionally remain alive
    // until process exit so late StorageView destructors keep a valid context.
    void destroy_context();

    std::string get_device_name(int device_index = -1);
    std::string get_backend_name(int device_index = -1);
    bool mayiuse_float16(int device_index = -1);
    bool mayiuse_bfloat16(int device_index = -1);
    bool mayiuse_int8(int device_index = -1);

    // Copies between allocations on two SYCL devices. Copies inside one
    // context remain asynchronous unless synchronous is true. Cross-context
    // copies use a synchronized host staging buffer.
    void copy(void* destination,
              int destination_device_index,
              const void* source,
              int source_device_index,
              size_t size,
              bool synchronous = false);

  }
}
