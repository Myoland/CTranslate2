#include "utils.h"

#include "random.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ctranslate2 {
  namespace sycl_backend {

    IntelGpuModel intel_gpu_model_from_device_id(
      const uint32_t device_id,
      const uint32_t execution_units) {
      switch (device_id) {
      case 0xe20b:
        return IntelGpuModel::ArcB580;
      case 0xb080:
      case 0xb082:
      case 0xb084:
      case 0xb086:
        return IntelGpuModel::ArcB390;
      case 0x64a0:
        // Lunar Lake shares this PCI ID between Arc 130V and 140V. Only the
        // full 64-vector-engine 140V was measured for these tuned paths.
        return execution_units >= 64
                 ? IntelGpuModel::Arc140V
                 : IntelGpuModel::Other;
      default:
        return IntelGpuModel::Other;
      }
    }

    namespace {

      constexpr uint32_t intel_vendor_id = 0x8086;

      IntelGpuModel detect_intel_gpu_model(const ::sycl::device& device) {
        try {
          if (device.has(::sycl::aspect::ext_intel_device_id)) {
            const uint32_t device_id = device.get_info<
              ::sycl::ext::intel::info::device::device_id>();
            uint32_t execution_units = 0;
            if (device.has(::sycl::aspect::ext_intel_gpu_eu_count)) {
              execution_units = device.get_info<
                ::sycl::ext::intel::info::device::gpu_eu_count>();
            }
            const IntelGpuModel model
              = intel_gpu_model_from_device_id(device_id, execution_units);
            if (model != IntelGpuModel::Other)
              return model;
          }
        } catch (const ::sycl::exception&) {
          // Preserve generic dispatch when an older runtime does not expose
          // the optional Intel device ID query.
        }

        // Keep compatibility with runtimes that identified B580 by name
        // before exposing ext_intel_device_id. B390 names are not stable and
        // therefore intentionally rely on the exact PCI ID mapping above.
        const std::string name
          = device.get_info<::sycl::info::device::name>();
        return name.find("B580") != std::string::npos
                 ? IntelGpuModel::ArcB580
                 : IntelGpuModel::Other;
      }

      class AsyncErrors {
      public:
        void add(std::exception_ptr error) {
          std::lock_guard<std::mutex> lock(_mutex);
          _errors.emplace_back(std::move(error));
        }

        void throw_pending() {
          std::exception_ptr error;
          {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_errors.empty())
              return;
            error = _errors.front();
            _errors.clear();
          }
          std::rethrow_exception(error);
        }

      private:
        std::mutex _mutex;
        std::vector<std::exception_ptr> _errors;
      };

      class QueueState {
      public:
        QueueState(const ::sycl::context& context, const ::sycl::device& device)
          : _errors(std::make_shared<AsyncErrors>())
          , _queue(context,
                   device,
                   [errors = _errors](const ::sycl::exception_list& exceptions) {
                     for (const std::exception_ptr& exception : exceptions)
                       errors->add(exception);
                   },
                   ::sycl::property_list{::sycl::property::queue::in_order()}) {
        }

        ~QueueState() {
          // Destructors must not throw. Public synchronization points surface
          // asynchronous failures while the queue is in active use.
          try {
            _queue.wait_and_throw();
          } catch (...) {
          }
        }

        ::sycl::queue& queue() {
          _errors->throw_pending();
          return _queue;
        }

        void wait() {
          try {
            _queue.wait_and_throw();
          } catch (...) {
            _errors->add(std::current_exception());
          }
          _errors->throw_pending();
        }

        void check_async_errors() {
          _errors->throw_pending();
        }

      private:
        std::shared_ptr<AsyncErrors> _errors;
        ::sycl::queue _queue;
      };

      struct DeviceEntry {
        explicit DeviceEntry(::sycl::device selected_device)
          : device(std::move(selected_device))
          , context(device)
          , model(detect_intel_gpu_model(device)) {
        }

        ::sycl::device device;
        ::sycl::context context;
        IntelGpuModel model;
      };

      class Registry {
      public:
        Registry() {
          try {
            for (::sycl::device device
                 : ::sycl::device::get_devices(::sycl::info::device_type::gpu)) {
              try {
                if (device.get_backend() != ::sycl::backend::ext_oneapi_level_zero)
                  continue;
                if (device.get_info<::sycl::info::device::vendor_id>() != intel_vendor_id)
                  continue;
                if (!device.has(::sycl::aspect::usm_device_allocations))
                  continue;
                _devices.emplace_back(std::move(device));
              } catch (const ::sycl::exception& error) {
                // A broken device should not hide other usable Intel GPUs.
                if (_initialization_error.empty())
                  _initialization_error = error.what();
              }
            }
          } catch (const std::exception& error) {
            _initialization_error = error.what();
            _devices.clear();
          }
          _queues.resize(_devices.size());
        }

        int size() const {
          return static_cast<int>(_devices.size());
        }

        int resolve_index(int device_index) const {
          if (device_index < 0)
            device_index = current_device_index;
          if (device_index < 0 || device_index >= size()) {
            std::string message = "Invalid SYCL device index " + std::to_string(device_index)
                                  + ": " + std::to_string(size())
                                  + " compatible Intel Level Zero device(s) are visible";
            if (!_initialization_error.empty())
              message += " (SYCL initialization failed: " + _initialization_error + ")";
            throw std::invalid_argument(message);
          }
          return device_index;
        }

        const DeviceEntry& get(int device_index) const {
          return _devices[resolve_index(device_index)];
        }

        void register_queue(int device_index, const std::shared_ptr<QueueState>& queue) {
          std::lock_guard<std::mutex> lock(_queues_mutex);
          auto& queues = _queues[device_index];
          queues.erase(std::remove_if(queues.begin(),
                                      queues.end(),
                                      [](const std::weak_ptr<QueueState>& entry) {
                                        return entry.expired();
                                      }),
                       queues.end());
          queues.emplace_back(queue);
        }

        std::vector<std::shared_ptr<QueueState>> queues(int device_index) {
          std::vector<std::shared_ptr<QueueState>> result;
          std::lock_guard<std::mutex> lock(_queues_mutex);
          auto& queues = _queues[resolve_index(device_index)];
          auto it = queues.begin();
          while (it != queues.end()) {
            if (auto queue = it->lock()) {
              result.emplace_back(std::move(queue));
              ++it;
            } else {
              it = queues.erase(it);
            }
          }
          return result;
        }

        static thread_local int current_device_index;

      private:
        std::vector<DeviceEntry> _devices;
        std::string _initialization_error;
        std::mutex _queues_mutex;
        std::vector<std::vector<std::weak_ptr<QueueState>>> _queues;
      };

      thread_local int Registry::current_device_index = 0;

      // Keep the registry alive until process exit. In particular, static and
      // thread-local StorageView destructors may still need its SYCL contexts.
      Registry& registry() {
        static Registry* instance = new Registry;
        return *instance;
      }

      std::shared_ptr<QueueState>& get_queue_state(int device_index) {
        const int resolved_index = registry().resolve_index(device_index);
        static thread_local std::vector<std::shared_ptr<QueueState>> queues;
        if (queues.size() < static_cast<size_t>(registry().size()))
          queues.resize(registry().size());
        if (!queues[resolved_index]) {
          const auto& entry = registry().get(resolved_index);
          queues[resolved_index] = std::make_shared<QueueState>(entry.context, entry.device);
          registry().register_queue(resolved_index, queues[resolved_index]);
        }
        return queues[resolved_index];
      }

      bool name_contains(const std::string& name, const std::string& token) {
        return name.find(token) != std::string::npos;
      }

    }

    int get_device_count() {
      return registry().size();
    }

    bool has_gpu() {
      return get_device_count() > 0;
    }

    int get_device_index() {
      // Match CUDA's behavior: selecting a device requires it to exist.
      return registry().resolve_index(Registry::current_device_index);
    }

    void set_device_index(int index) {
      Registry::current_device_index = registry().resolve_index(index);
    }

    const ::sycl::device& get_device(int device_index) {
      return registry().get(device_index).device;
    }

    const ::sycl::context& get_context(int device_index) {
      return registry().get(device_index).context;
    }

    IntelGpuModel get_intel_gpu_model(int device_index) {
      return registry().get(device_index).model;
    }

    ::sycl::queue& get_queue(int device_index) {
      return get_queue_state(device_index)->queue();
    }

    void synchronize(int device_index) {
      const int resolved_index = registry().resolve_index(device_index);
      for (const auto& queue : registry().queues(resolved_index))
        queue->wait();
    }

    void synchronize_queue() {
      get_queue_state(-1)->wait();
    }

    void check_async_errors() {
      get_queue_state(-1)->check_async_errors();
    }

    void destroy_context() {
      reset_random_generator();
      for (int device_index = 0; device_index < get_device_count(); ++device_index)
        synchronize(device_index);
    }

    std::string get_device_name(int device_index) {
      return get_device(device_index).get_info<::sycl::info::device::name>();
    }

    std::string get_backend_name(int device_index) {
      // Calling get_device validates the index and any deferred initialization
      // error. The registry only admits Level Zero devices.
      (void)get_device(device_index);
      return "level_zero";
    }

    bool mayiuse_float16(int device_index) {
      return get_device(device_index).has(::sycl::aspect::fp16);
    }

    bool mayiuse_bfloat16(int device_index) {
      // oneMKL exposes optimized BF16 GEMM on Intel Arc and Data Center GPUs.
      // Be conservative for older integrated GPUs where bfloat16 may merely be
      // emulated by the compiler extension.
      const std::string name = get_device_name(device_index);
      return name_contains(name, "Arc")
             || name_contains(name, "Data Center GPU")
             || name_contains(name, "Flex")
             || name_contains(name, "Max");
    }

    bool mayiuse_int8(int device_index) {
      // All devices admitted by this Intel-only registry support integer
      // kernels and oneMKL signed INT8 GEMM.
      (void)get_device(device_index);
      return true;
    }

    void copy(void* destination,
              int destination_device_index,
              const void* source,
              int source_device_index,
              size_t size,
              bool synchronous) {
      if (size == 0)
        return;

      destination_device_index = registry().resolve_index(destination_device_index);
      source_device_index = registry().resolve_index(source_device_index);

      if (destination_device_index == source_device_index) {
        ::sycl::event event = get_queue(destination_device_index).memcpy(destination,
                                                                         source,
                                                                         size);
        if (synchronous) {
          event.wait_and_throw();
          get_queue_state(destination_device_index)->check_async_errors();
        }
        return;
      }

      // Contexts are per device. Use an explicit host staging allocation for
      // cross-device copies instead of relying on implementation-specific peer
      // access between contexts.
      synchronize(source_device_index);
      std::vector<unsigned char> staging(size);
      get_queue(source_device_index).memcpy(staging.data(), source, size).wait_and_throw();
      get_queue(destination_device_index).memcpy(destination, staging.data(), size).wait_and_throw();
      get_queue_state(source_device_index)->check_async_errors();
      get_queue_state(destination_device_index)->check_async_errors();
    }

  }
}
