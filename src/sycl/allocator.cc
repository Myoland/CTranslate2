#include "ctranslate2/allocator.h"

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sycl/utils.h"

namespace ctranslate2 {
  namespace sycl_backend {
    namespace {

      struct CachedBlock {
        void* pointer = nullptr;
        size_t size = 0;
      };

      struct DeviceCache {
        std::mutex mutex;
        // Serializes the device synchronization used to retire pending
        // allocations. Normal allocation and deallocation only take mutex.
        std::mutex retirement_mutex;
        std::unordered_map<void*, size_t> live_allocations;
        std::multimap<size_t, CachedBlock> cached_allocations;
        std::vector<CachedBlock> pending_allocations;
        size_t cached_bytes = 0;
        size_t pending_bytes = 0;
        size_t retiring_bytes = 0;
      };

      size_t get_max_cached_bytes() {
        static const size_t value = []() {
          constexpr size_t default_value = 512ull * 1024 * 1024;
          const char* configured_value = std::getenv("CT2_SYCL_CACHING_ALLOCATOR_MAX_BYTES");
          if (!configured_value)
            return default_value;

          const std::string value_string(configured_value);
          const auto is_decimal_digit = [](char value) {
            return value >= '0' && value <= '9';
          };
          if (value_string.empty()
              || !std::all_of(value_string.begin(), value_string.end(), is_decimal_digit)) {
            throw std::invalid_argument(
              "CT2_SYCL_CACHING_ALLOCATOR_MAX_BYTES must be an unsigned integer");
          }

          try {
            const unsigned long long value = std::stoull(value_string);
            if (value > std::numeric_limits<size_t>::max())
              throw std::out_of_range("allocator cache limit exceeds size_t");
            return static_cast<size_t>(value);
          } catch (const std::exception&) {
            throw std::invalid_argument(
              "CT2_SYCL_CACHING_ALLOCATOR_MAX_BYTES must be an unsigned integer");
          }
        }();
        return value;
      }

      class SyclCachingAllocator : public Allocator {
      public:
        SyclCachingAllocator() {
          // Validate allocator configuration before any StorageView owns a
          // pointer, so a malformed value cannot first surface in a destructor.
          (void)get_max_cached_bytes();
        }

        void* allocate(size_t size, int device_index) override {
          device_index = resolve_device_index(device_index);
          DeviceCache& cache = get_cache(device_index);

          if (void* pointer = take_cached_allocation(cache, size))
            return pointer;

          void* pointer = try_allocate(size, device_index);
          if (!pointer) {
            // Pending allocations may account for the memory pressure, but
            // cannot be reused until all queues that could have consumed them
            // are synchronized.
            reclaim_pending(device_index);
            if (void* cached_pointer = take_cached_allocation(cache, size))
              return cached_pointer;
          }
          if (!pointer) {
            clear_cache(device_index);
            pointer = try_allocate(size, device_index);
          }
          if (!pointer) {
            throw std::runtime_error("SYCL device allocation of " + std::to_string(size)
                                     + " bytes failed on " + get_device_name(device_index));
          }

          {
            std::lock_guard<std::mutex> lock(cache.mutex);
            cache.live_allocations.emplace(pointer, size);
          }
          return pointer;
        }

        void free(void* pointer, int device_index) override {
          if (!pointer)
            return;

          device_index = resolve_device_index(device_index);
          DeviceCache& cache = get_cache(device_index);
          size_t size = 0;
          bool trim_cache = false;
          {
            std::lock_guard<std::mutex> lock(cache.mutex);
            const auto live_it = cache.live_allocations.find(pointer);
            if (live_it == cache.live_allocations.end())
              return;
            size = live_it->second;
            cache.live_allocations.erase(live_it);
            cache.pending_allocations.push_back({pointer, size});
            cache.pending_bytes += size;
            trim_cache = pooled_bytes(cache) > get_max_cached_bytes();
          }

          if (trim_cache)
            trim_cache_to_limit(device_index);
        }

        void clear_cache() override {
          for (int device_index = 0; device_index < get_device_count(); ++device_index)
            clear_cache(device_index);
        }

      private:
        static int resolve_device_index(int device_index) {
          if (device_index < 0)
            return get_device_index();
          (void)get_device(device_index);  // Validate the explicit index.
          return device_index;
        }

        DeviceCache& get_cache(int device_index) {
          std::lock_guard<std::mutex> lock(_caches_mutex);
          if (_caches.size() < static_cast<size_t>(get_device_count()))
            _caches.resize(get_device_count());
          if (!_caches[device_index])
            _caches[device_index] = std::make_unique<DeviceCache>();
          return *_caches[device_index];
        }

        static void* try_allocate(size_t size, int device_index) {
          try {
            return ::sycl::malloc_device(size,
                                         get_device(device_index),
                                         get_context(device_index));
          } catch (const ::sycl::exception&) {
            return nullptr;
          }
        }

        static size_t pooled_bytes(const DeviceCache& cache) {
          return cache.cached_bytes + cache.pending_bytes + cache.retiring_bytes;
        }

        static void* take_cached_allocation(DeviceCache& cache, size_t size) {
          std::lock_guard<std::mutex> lock(cache.mutex);
          auto cached_it = cache.cached_allocations.lower_bound(size);
          if (cached_it == cache.cached_allocations.end())
            return nullptr;

          CachedBlock block = std::move(cached_it->second);
          cache.cached_bytes -= block.size;
          cache.cached_allocations.erase(cached_it);
          cache.live_allocations.emplace(block.pointer, block.size);
          return block.pointer;
        }

        static std::vector<CachedBlock> take_pending_allocations(DeviceCache& cache,
                                                                 size_t& bytes) {
          std::vector<CachedBlock> blocks;
          {
            std::lock_guard<std::mutex> lock(cache.mutex);
            blocks.swap(cache.pending_allocations);
            bytes = cache.pending_bytes;
            cache.pending_bytes = 0;
            cache.retiring_bytes += bytes;
          }
          return blocks;
        }

        static void restore_pending_allocations(DeviceCache& cache,
                                                std::vector<CachedBlock>& blocks,
                                                size_t bytes) {
          std::lock_guard<std::mutex> lock(cache.mutex);
          cache.pending_allocations.insert(cache.pending_allocations.end(),
                                           std::make_move_iterator(blocks.begin()),
                                           std::make_move_iterator(blocks.end()));
          cache.pending_bytes += bytes;
          cache.retiring_bytes -= bytes;
        }

        static void promote_pending_allocations(DeviceCache& cache,
                                                std::vector<CachedBlock>& blocks,
                                                size_t bytes) {
          std::lock_guard<std::mutex> lock(cache.mutex);
          for (CachedBlock& block : blocks)
            cache.cached_allocations.emplace(block.size, std::move(block));
          cache.cached_bytes += bytes;
          cache.retiring_bytes -= bytes;
        }

        // The allocator does not know which thread-local queues consumed a
        // pointer. A block freed on thread B may still be referenced by work on
        // thread A, so it is unsafe to retire it with a barrier on B's queue.
        // Synchronizing all queues for the device establishes a retirement
        // point for a snapshot of pending blocks. Frees concurrent with the
        // synchronization remain pending for the next retirement point.
        void reclaim_pending(int device_index) {
          DeviceCache& cache = get_cache(device_index);
          std::lock_guard<std::mutex> retirement_lock(cache.retirement_mutex);

          size_t bytes = 0;
          std::vector<CachedBlock> blocks = take_pending_allocations(cache, bytes);
          if (blocks.empty())
            return;

          try {
            synchronize(device_index);
          } catch (...) {
            restore_pending_allocations(cache, blocks, bytes);
            throw;
          }

          promote_pending_allocations(cache, blocks, bytes);
        }

        void trim_cache_to_limit(int device_index) {
          DeviceCache& cache = get_cache(device_index);
          std::lock_guard<std::mutex> retirement_lock(cache.retirement_mutex);

          size_t pending_bytes = 0;
          std::vector<CachedBlock> pending
            = take_pending_allocations(cache, pending_bytes);
          if (!pending.empty()) {
            try {
              synchronize(device_index);
            } catch (...) {
              restore_pending_allocations(cache, pending, pending_bytes);
              throw;
            }
            promote_pending_allocations(cache, pending, pending_bytes);
          }

          std::vector<CachedBlock> blocks_to_free;
          {
            std::lock_guard<std::mutex> lock(cache.mutex);
            while (pooled_bytes(cache) > get_max_cached_bytes()
                   && !cache.cached_allocations.empty()) {
              auto it = std::prev(cache.cached_allocations.end());
              cache.cached_bytes -= it->second.size;
              blocks_to_free.emplace_back(std::move(it->second));
              cache.cached_allocations.erase(it);
            }
          }

          // All cached blocks crossed a device-wide retirement point before
          // they became reusable, so releasing them no longer needs a wait.
          for (const CachedBlock& block : blocks_to_free)
            ::sycl::free(block.pointer, get_context(device_index));
        }

        void clear_cache(int device_index) {
          DeviceCache& cache = get_cache(device_index);
          std::lock_guard<std::mutex> retirement_lock(cache.retirement_mutex);

          std::vector<CachedBlock> blocks;
          size_t pending_bytes = 0;
          size_t retiring_bytes = 0;
          {
            std::lock_guard<std::mutex> lock(cache.mutex);
            blocks.reserve(cache.cached_allocations.size()
                           + cache.pending_allocations.size());
            for (auto& entry : cache.cached_allocations)
              blocks.emplace_back(std::move(entry.second));
            for (CachedBlock& block : cache.pending_allocations)
              blocks.emplace_back(std::move(block));
            pending_bytes = cache.pending_bytes;
            retiring_bytes = cache.cached_bytes + pending_bytes;
            cache.cached_allocations.clear();
            cache.pending_allocations.clear();
            cache.cached_bytes = 0;
            cache.pending_bytes = 0;
            cache.retiring_bytes += retiring_bytes;
          }

          if (blocks.empty())
            return;

          try {
            if (pending_bytes != 0)
              synchronize(device_index);
          } catch (...) {
            std::lock_guard<std::mutex> lock(cache.mutex);
            for (CachedBlock& block : blocks) {
              cache.pending_bytes += block.size;
              cache.pending_allocations.emplace_back(std::move(block));
            }
            cache.retiring_bytes -= retiring_bytes;
            throw;
          }

          for (const CachedBlock& block : blocks)
            ::sycl::free(block.pointer, get_context(device_index));

          {
            std::lock_guard<std::mutex> lock(cache.mutex);
            cache.retiring_bytes -= retiring_bytes;
          }
        }

        std::mutex _caches_mutex;
        std::vector<std::unique_ptr<DeviceCache>> _caches;
      };

    }
  }

  template<>
  Allocator& get_allocator<Device::SYCL>() {
    static sycl_backend::SyclCachingAllocator allocator;
    return allocator;
  }

}
