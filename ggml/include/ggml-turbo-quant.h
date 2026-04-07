/**
 * ggml-turbo-quant.h -- quant.cpp 1-bit KV cache quantization for llama.cpp
 *
 * Apache 2.0 License, QuantumAI Inc.
 *
 * Self-contained implementation of quant.cpp 1-bit KV cache compression.
 * Algorithm: L2-normalize -> Random Hadamard Transform -> sign extraction.
 * Attention: XOR + popcount Hamming distance -> inner product estimator.
 *
 * Reference: quant.cpp (arXiv 2504.19874)
 *   - 1-bit per dimension with RHT decorrelation
 *   - Theoretical attention cosine similarity: 2/pi ~ 0.637
 *   - Compression: 24 bytes per 128 elements (1.5 bpw including metadata)
 *
 * Usage in llama.cpp:
 *   --cache-type-k tq_kv_1b   (for key cache)
 *   --cache-type-v tq_kv_1b   (for value cache, though key-only is recommended)
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

/* GGML_RESTRICT definition (use ggml.h if available, otherwise define locally) */
#ifndef GGML_RESTRICT
#  if defined(__GNUC__) || defined(__clang__)
#    define GGML_RESTRICT __restrict__
#  elif defined(_MSC_VER)
#    define GGML_RESTRICT __restrict
#  elif defined(__CUDACC__)
#    define GGML_RESTRICT __restrict
#  else
#    define GGML_RESTRICT
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Block definition: quant.cpp 1-bit KV cache
 *
 * 24 bytes per 128 elements = 1.5 bits per element (with metadata)
 * Pure sign bits = 1.0 bpw, metadata overhead = 0.5 bpw
 *
 * Layout:
 *   norm     (2B) - L2 norm of original vector, stored as FP16
 *   _pad     (2B) - alignment padding (reserved for future use)
 *   rht_seed (4B) - RHT random seed for inverse transform
 *   signs   (16B) - 128 sign bits, LSB-first packing
 *
 * Total: 24 bytes per 128 elements
 * Compression vs FP16: 256 bytes / 24 bytes = 10.7x
 * Compression vs FP32: 512 bytes / 24 bytes = 21.3x
 * ============================================================ */

#define TQ_KV_1B_BLOCK_SIZE 128

typedef struct {
    uint16_t norm;                              /* L2 norm in FP16               */
    uint16_t _pad;                              /* alignment padding             */
    uint32_t rht_seed;                          /* RHT seed for inverse          */
    uint8_t  signs[TQ_KV_1B_BLOCK_SIZE / 8];   /* 128 sign bits = 16 bytes      */
} block_tq_kv_1b;

/* Compile-time size check: 2 + 2 + 4 + 16 = 24 bytes */
typedef char tq_check_block_size[(sizeof(block_tq_kv_1b) == 24) ? 1 : -1];

/* ============================================================
 * Block definition: quant.cpp Uniform 4-bit KV cache
 *
 * 68 bytes per 128 elements = 4.25 bits per element (with metadata)
 * 4-bit quantization with min-max uniform scalar quantization
 *
 * Layout:
 *   scale      (2B) - (max - min) / 15, stored as FP16
 *   zero_point (2B) - minimum value, stored as FP16
 *   qs        (64B) - 4-bit quantized values, 2 per byte, LSB-first
 *
 * Total: 68 bytes per 128 elements
 * Compression vs FP16: 256 bytes / 68 bytes = 3.76x
 * Compression vs FP32: 512 bytes / 68 bytes = 7.53x
 * ============================================================ */

#define TQ_KV_4B_UNIFORM_BLOCK_SIZE 128

typedef struct {
    uint16_t scale;                              /* FP16 scale = (max - min) / 15    */
    uint16_t zero_point;                         /* FP16 minimum value               */
    uint8_t  qs[TQ_KV_4B_UNIFORM_BLOCK_SIZE / 2]; /* 4-bit values, 2 per byte (64B)  */
} block_tq_kv_4b_uniform;

/* Compile-time size check: 2 + 2 + 64 = 68 bytes */
typedef char tq_check_block_size_4b[(sizeof(block_tq_kv_4b_uniform) == 68) ? 1 : -1];

/* ============================================================
 * Public API (matches llama.cpp quantize/dequantize convention)
 *
 * k: number of elements (must be multiple of block size)
 * ============================================================ */

/**
 * Quantize a row of float values to 1-bit quant.cpp KV blocks.
 *
 * Pipeline: L2-normalize -> RHT (Walsh-Hadamard + random signs) -> sign extraction.
 *
 * @param x   Input float array (k elements)
 * @param y   Output block array (k / TQ_KV_1B_BLOCK_SIZE blocks)
 * @param k   Number of elements (must be multiple of 128)
 */
void quantize_row_tq_kv_1b_ref(const float * x, block_tq_kv_1b * y, int64_t k);

/**
 * Dequantize 1-bit quant.cpp KV blocks back to float.
 *
 * Pipeline: sign -> scale by sqrt(2/pi)/sqrt(dim) -> inverse RHT -> scale by norm.
 * Note: This is a rough reconstruction. The real value of 1-bit is in Hamming attention.
 *
 * @param x   Input block array
 * @param y   Output float array (k elements)
 * @param k   Number of elements (must be multiple of 128)
 */
void dequantize_row_tq_kv_1b(const block_tq_kv_1b * x, float * y, int64_t k);

/**
 * Vector dot product between TQ_KV_1B (quantized key) and F32 (query).
 * Used in attention mechanism for KV cache with 1-bit quantization.
 *
 * This is a dequantize-based implementation:
 *   1. Dequantize each KV block using inverse RHT
 *   2. Compute dot product with query
 *
 * Note: For production use, consider tq_kv_1b_attention which uses
 * XOR+popcount for O(1) per-key attention computation.
 *
 * @param n     Number of elements
 * @param s     Output scalar result
 * @param bs    Stride for s (usually 0)
 * @param vx    TQ_KV_1B blocks (quantized key)
 * @param bx    Stride for vx (usually 0)
 * @param vy    F32 query vector
 * @param by    Stride for vy (usually 0)
 * @param nrc   Number of rows to compute (must be 1)
 */
void ggml_vec_dot_tq_kv_1b_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                const void * GGML_RESTRICT vx, size_t bx,
                                const void * GGML_RESTRICT vy, size_t by, int nrc);

/**
 * Compute attention scores between a query and quantized KV cache.
 *
 * Uses XOR + popcount Hamming distance for ultra-fast attention:
 *   score = q_norm * k_norm * sqrt(pi/2) / dim * (2*agree - dim)
 *
 * @param query     Float query vector (head_dim elements)
 * @param kv_cache  Array of quantized key blocks (seq_len blocks)
 * @param scores    Output attention scores (seq_len elements)
 * @param seq_len   Number of keys in the cache
 * @param head_dim  Dimension of each head (must be <= 128)
 */
void tq_kv_1b_attention(const float * query, const block_tq_kv_1b * kv_cache,
                         float * scores, int seq_len, int head_dim);

/**
 * Quantize a row of float values to 4-bit uniform KV blocks.
 *
 * Pipeline: find min/max -> linear map to 16 levels -> 4-bit packing.
 *
 * @param x   Input float array (k elements)
 * @param y   Output block array (k / TQ_KV_4B_UNIFORM_BLOCK_SIZE blocks)
 * @param k   Number of elements (must be multiple of 128)
 */
void quantize_row_tq_kv_4b_uniform_ref(const float * x, block_tq_kv_4b_uniform * y, int64_t k);

/**
 * Dequantize 4-bit uniform KV blocks back to float.
 *
 * Pipeline: extract 4-bit index -> scale and shift -> reconstruct value.
 *
 * @param x   Input block array
 * @param y   Output float array (k elements)
 * @param k   Number of elements (must be multiple of 128)
 */
void dequantize_row_tq_kv_4b_uniform(const block_tq_kv_4b_uniform * x, float * y, int64_t k);

/**
 * Vector dot product between TQ_KV_4B_UNIFORM (quantized key) and F32 (query).
 * Used in attention mechanism for KV cache with 4-bit quantization.
 *
 * This is a dequantize-based implementation:
 *   1. Dequantize each KV block
 *   2. Compute dot product with query
 *
 * @param n     Number of elements
 * @param s     Output scalar result
 * @param bs    Stride for s (usually 0)
 * @param vx    TQ_KV_4B_UNIFORM blocks (quantized key)
 * @param bx    Stride for vx (usually 0)
 * @param vy    F32 query vector
 * @param by    Stride for vy (usually 0)
 * @param nrc   Number of rows to compute (must be 1)
 */
void ggml_vec_dot_tq_kv_4b_uniform_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                        const void * GGML_RESTRICT vx, size_t bx,
                                        const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
