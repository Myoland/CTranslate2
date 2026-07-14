#include "sycl/random.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <oneapi/mkl/rng.hpp>

#include "ctranslate2/random.h"
#include "sycl/utils.h"

namespace ctranslate2 {
  namespace sycl_backend {
    namespace {

      struct RandomGenerator {
        RandomGenerator(::sycl::queue& queue, const std::uint64_t seed, const int device_index)
          : engine(queue, seed)
          , device_index(device_index) {
        }

        oneapi::mkl::rng::philox4x32x10 engine;
        int device_index;
      };

      thread_local std::vector<std::unique_ptr<RandomGenerator>> generators;

    }

    void fill_uniform(float* output, const size_t size) {
      if (size == 0)
        return;

      const int device_index = get_device_index();
      if (generators.size() <= static_cast<size_t>(device_index))
        generators.resize(device_index + 1);
      auto& generator = generators[device_index];
      if (!generator) {
        generator = std::make_unique<RandomGenerator>(
          get_queue(), static_cast<std::uint64_t>(get_random_seed()), device_index);
      }

      oneapi::mkl::rng::uniform<float> distribution(0.f, 1.f);
      oneapi::mkl::rng::generate(distribution,
                                 generator->engine,
                                 static_cast<std::int64_t>(size),
                                 output);
    }

    void reset_random_generator() {
      for (auto& generator : generators) {
        if (generator)
          synchronize(generator->device_index);
      }
      generators.clear();
    }

  }
}
