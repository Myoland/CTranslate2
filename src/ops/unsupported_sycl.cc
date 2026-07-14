#include "ctranslate2/ops/awq/dequantize_awq.h"
#include "ctranslate2/ops/awq/gemm.h"
#include "ctranslate2/ops/awq/gemv.h"
#include "ctranslate2/ops/flash_attention.h"
#include "ctranslate2/ops/nccl_ops.h"

#include <stdexcept>

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <>
    void FlashAttention::compute<Device::SYCL>(StorageView&,
                                                StorageView&,
                                                StorageView&,
                                                StorageView&,
                                                StorageView*,
                                                StorageView*,
                                                StorageView*,
                                                bool,
                                                StorageView*,
                                                StorageView*,
                                                const bool,
                                                StorageView*,
                                                dim_t) const {
      throw std::runtime_error(
        "FlashAttention 2 is not supported by the SYCL backend; disable flash_attention");
    }

    template <Device D, typename InT, typename OutT>
    void DequantizeAwq::dequantize(const StorageView&,
                                   const StorageView&,
                                   const StorageView&,
                                   StorageView&) const {
      throw std::runtime_error("AWQ INT4 is not supported by the SYCL backend");
    }

    template void DequantizeAwq::dequantize<Device::SYCL, int, float16_t>(
      const StorageView&, const StorageView&, const StorageView&, StorageView&) const;

    template <Device D, typename In, typename Out>
    void GemmAwq::compute(const StorageView&,
                          const StorageView&,
                          const StorageView&,
                          const StorageView&,
                          StorageView&) const {
      throw std::runtime_error("AWQ INT4 is not supported by the SYCL backend");
    }

    template void GemmAwq::compute<Device::SYCL, float16_t, int>(
      const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const;

    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv(const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               const StorageView&,
                               StorageView&) const {
      throw std::runtime_error("AWQ INT4 is not supported by the SYCL backend");
    }

    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv2(const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                const StorageView&,
                                StorageView&) const {
      throw std::runtime_error("AWQ INT4 is not supported by the SYCL backend");
    }

    template void GemvAwq::compute_gemv<Device::SYCL, float16_t, int>(
      const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const;
    template void GemvAwq::compute_gemv2<Device::SYCL, float16_t, int>(
      const StorageView&, const StorageView&, const StorageView&, const StorageView&, StorageView&) const;

    template <Device D, typename T>
    void ReduceAll::compute(const StorageView&, StorageView&) const {
      throw std::runtime_error("Tensor parallel collectives are not supported by the SYCL backend");
    }

    template <Device D, typename T>
    void GatherAll::compute(const StorageView&, StorageView&) const {
      throw std::runtime_error("Tensor parallel collectives are not supported by the SYCL backend");
    }

#define DECLARE_COLLECTIVE(T)                                          \
    template void ReduceAll::compute<Device::SYCL, T>(                 \
      const StorageView&, StorageView&) const;                         \
    template void GatherAll::compute<Device::SYCL, T>(                 \
      const StorageView&, StorageView&) const;

    DECLARE_ALL_TYPES(DECLARE_COLLECTIVE)

#undef DECLARE_COLLECTIVE

  }
}
