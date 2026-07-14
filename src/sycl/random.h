#pragma once

#include <cstddef>

namespace ctranslate2 {
  namespace sycl_backend {

    // Fills device USM with values sampled uniformly from [0, 1).  The engine
    // is thread-local and advances between calls. Each device has an independent
    // engine so switching devices does not reset either random sequence.
    void fill_uniform(float* output, size_t size);

    // Releases all generators associated with the calling thread. Runtime
    // teardown should call this before destroying its queue/context.
    void reset_random_generator();

  }
}
