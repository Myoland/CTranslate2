#pragma once

#include "ctranslate2/types.h"
#include "sycl/device_info.h"

namespace ctranslate2 {
  namespace sycl_backend {

    struct FusedSingleQueryAttentionFP16Config {
      dim_t batch_size;
      dim_t num_heads;
      dim_t query_length;
      dim_t key_length;
      dim_t head_dim;
      bool has_mask = false;
      bool returns_attention = false;
      bool has_relative_position = false;
      bool has_alibi = false;
    };

    // Returns the measured product-specific profitability decision without
    // querying a device. Decoder batches and beams are flattened into rows.
    bool is_fused_attention_profitable(IntelGpuModel model,
                                       dim_t batch_size,
                                       dim_t key_length);

    // Returns whether the selected SYCL device can run the specialized
    // single-query attention kernel for this shape.  Inputs use the contiguous
    // [batch, heads, length, 64] layout and each batch has its own K/V cache.
    // On tuned Arc products, the predicate includes a product-specific
    // profitability boundary: up to 64 independent Whisper decoder rows with
    // a row-dependent key-length limit. Other SYCL GPUs retain the original
    // exact beam-5 dispatch. All accepted shapes use 20 heads and no attention
    // features requiring extra outputs or score modifications. Decoder batches
    // and beams are flattened into these rows; the kernel does not share state
    // between them.
    bool supports_fused_single_query_attention_fp16(
      const FusedSingleQueryAttentionFP16Config& config);

    // Computes scaled dot-product attention for one query per batch/head:
    //
    //   queries: [batch, heads, 1, head_dim]
    //   keys:    [batch, heads, key_length, head_dim]
    //   values:  [batch, heads, key_length, head_dim]
    //   output:  [batch, heads, 1, head_dim]
    //
    // Scores and normalized probabilities are rounded to float16 before their
    // next use, matching the intermediate types of the unfused FP16 path.
    // Work is enqueued asynchronously on the current in-order SYCL queue.
    void fused_single_query_attention_fp16(const float16_t* queries,
                                           const float16_t* keys,
                                           const float16_t* values,
                                           float16_t* output,
                                           dim_t batch_size,
                                           dim_t num_heads,
                                           dim_t key_length,
                                           dim_t head_dim,
                                           float scale);

  }
}
