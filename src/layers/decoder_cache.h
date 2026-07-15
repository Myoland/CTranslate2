#pragma once

#include <utility>

#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/ops/gather.h"

namespace ctranslate2 {
  namespace layers {
    namespace detail {

      inline constexpr const char* deferred_self_cache_indices_name
        = "__deferred_self_cache_indices";

      inline thread_local const StorageView* current_self_cache_indices = nullptr;

      class ScopedSelfCacheIndices {
      public:
        explicit ScopedSelfCacheIndices(const StorageView* indices)
          : _previous(current_self_cache_indices) {
          current_self_cache_indices = indices;
        }

        ~ScopedSelfCacheIndices() {
          current_self_cache_indices = _previous;
        }

        ScopedSelfCacheIndices(const ScopedSelfCacheIndices&) = delete;
        ScopedSelfCacheIndices& operator=(const ScopedSelfCacheIndices&) = delete;

      private:
        const StorageView* _previous;
      };

      inline const StorageView* get_self_cache_indices() {
        return current_self_cache_indices;
      }

      inline bool is_self_cache(const std::string& name) {
        return starts_with(name, "self_keys_")
          || starts_with(name, "self_values_");
      }

      inline void defer_cache_gather(DecoderState& state,
                                     StorageView indices,
                                     const char* state_name) {
        const auto existing = state.find(state_name);
        if (existing == state.end()) {
          state.emplace(state_name, std::move(indices));
        } else {
          if (indices.device() != existing->second.device())
            indices = indices.to(existing->second.device());
          StorageView composed_indices(DataType::INT32, indices.device());
          ops::Gather()(existing->second, indices, composed_indices);
          existing->second = std::move(composed_indices);
        }
      }

    }
  }
}
