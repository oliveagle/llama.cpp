/**
 * ggml-turbo-quant.c -- quant.cpp 1-bit KV cache quantization for llama.cpp
 *
 * Apache 2.0 License, QuantumAI Inc.
 *
 * Self-contained C99 implementation. No external dependencies beyond libc/libm.
 * Ported from quant.cpp src/core/tq_rht.c and src/core/tq_turbo_kv.c.
 *
 * Algorithm overview:
 *   Quantize:   L2-normalize -> RHT (random signs + Walsh-Hadamard) -> sign bits
 *   Dequantize: signs -> scale(sqrt(2/pi)/sqrt(dim)) -> inverse RHT -> scale(norm)
 *   Attention:  RHT(query) -> sign bits -> XOR + popcount -> Hamming score
 */

#include "ggml-turbo-quant.h"
#include <math.h>
#include <string.h>
#include <assert.h>

/* UNUSED macro for unused parameters (self-contained, no ggml dependency) */
#define UNUSED(x) (void)(x)

/* ============================================================
 * FP16 <-> FP32 conversion (self-contained, no ggml dependency)
 * ============================================================ */

static uint16_t tq_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float tq_fp16_to_fp32(uint16_t h) {
    union { float f; uint32_t u; } bits;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0)  { bits.u = sign; return bits.f; }
    if (exp == 31) { bits.u = sign | 0x7F800000 | (mant << 13); return bits.f; }
    exp = exp - 15 + 127;
    bits.u = sign | (exp << 23) | (mant << 13);
    return bits.f;
}

/* ============================================================
 * Constants
 * ============================================================ */

#define TQ_PI   3.14159265358979323846f
#define TQ_PI_2 1.5707963267948966f     /* pi/2 */

/* Default RHT seed -- all blocks use the same seed so we can
 * pre-rotate the query once and reuse across all keys. */
#define TQ_DEFAULT_SEED 0x12345678u

/* ============================================================
 * Random Hadamard Transform (RHT) -- self-contained port
 *
 * RHT = (1/sqrt(n)) * H * D where:
 *   D = diagonal random sign matrix (from seed)
 *   H = Walsh-Hadamard butterfly transform
 *
 * Properties:
 *   - RHT is orthogonal: preserves inner products
 *   - O(n log n) computation, no matrix storage
 *   - Decorrelates channels, making scalar quantization near-optimal
 *   - Self-inverse (up to scaling): H * H = n * I
 * ============================================================ */

/* Deterministic random sign from seed + index (Knuth multiplicative hash) */
static int tq_random_sign(uint32_t seed, int idx) {
    uint32_t h = seed ^ (uint32_t)idx;
    h = h * 2654435761u;
    return (h & 1) ? 1 : -1;
}

/* In-place Walsh-Hadamard Transform: O(n log n) butterfly.
 * n must be a power of 2. Self-inverse up to scaling: WHT(WHT(x)) = n * x. */
static void tq_walsh_hadamard(float * data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

/* Forward RHT: random sign flip -> WHT -> normalize by 1/sqrt(n) */
static void tq_rht_forward(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;

    /* Round down to nearest power of 2 */
    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;

    /* Step 1: Random sign flip (diagonal D matrix) */
    for (int i = 0; i < n2; i++) {
        data[i] *= (float)tq_random_sign(seed, i);
    }

    /* Step 2: Walsh-Hadamard butterfly */
    tq_walsh_hadamard(data, n2);

    /* Step 3: Normalize by 1/sqrt(n) for orthogonal transform */
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }
}

/* Inverse RHT: normalize -> WHT -> same sign flip.
 * Since H is self-inverse up to scaling and D*D = I:
 *   RHT     = (1/sqrt(n)) * H * D
 *   RHT^-1  = (1/sqrt(n)) * D * H  = D * (1/sqrt(n)) * H */
static void tq_rht_inverse(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;

    int n2 = 1;
    while (n2 * 2 <= n) n2 *= 2;

    /* Step 1: Normalize by 1/sqrt(n) */
    float scale = 1.0f / sqrtf((float)n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }

    /* Step 2: Walsh-Hadamard (self-inverse up to scaling) */
    tq_walsh_hadamard(data, n2);

    /* Step 3: Same random sign flip (D * D = I) */
    for (int i = 0; i < n2; i++) {
        data[i] *= (float)tq_random_sign(seed, i);
    }
}

/* ============================================================
 * Portable popcount (for platforms without __builtin_popcount)
 * ============================================================ */

static int tq_popcount8(uint8_t x) {
    int c = 0;
    while (x) { c++; x &= x - 1; }  /* Kernighan's bit trick */
    return c;
}

/* ============================================================
 * Quantize: float -> block_tq_kv_1b
 *
 * Pipeline per 128-element block:
 *   1. Compute L2 norm of block
 *   2. L2-normalize the block
 *   3. Apply RHT (random signs + Walsh-Hadamard + 1/sqrt(n))
 *   4. Extract sign bits (1 = positive, 0 = negative)
 *   5. Store norm (FP16) and RHT seed
 * ============================================================ */

void quantize_row_tq_kv_1b_ref(const float * x, block_tq_kv_1b * y, int64_t k) {
    const int block_size = TQ_KV_1B_BLOCK_SIZE;
    const int64_t num_blocks = k / block_size;

    for (int64_t b = 0; b < num_blocks; b++) {
        const float * src = x + b * block_size;
        block_tq_kv_1b * block = &y[b];

        /* Step 1: Compute L2 norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < block_size; i++) {
            norm_sq += src[i] * src[i];
        }
        float norm = sqrtf(norm_sq);
        block->norm = tq_fp32_to_fp16(norm);
        block->_pad = 0;

        /* Step 2: Normalize and copy to working buffer */
        float rotated[TQ_KV_1B_BLOCK_SIZE];
        float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
        for (int i = 0; i < block_size; i++) {
            rotated[i] = src[i] * inv_norm;
        }

        /* Step 3: Apply RHT */
        uint32_t seed = TQ_DEFAULT_SEED;
        block->rht_seed = seed;
        tq_rht_forward(rotated, block_size, seed);

        /* Step 4: Extract sign bits -- 1 bit per dimension, LSB-first */
        int sign_bytes = block_size / 8;
        memset(block->signs, 0, (size_t)sign_bytes);
        for (int i = 0; i < block_size; i++) {
            if (rotated[i] > 0.0f) {
                block->signs[i / 8] |= (uint8_t)(1 << (i % 8));
            }
        }
    }
}

/* ============================================================
 * Dequantize: block_tq_kv_1b -> float
 *
 * Pipeline per block:
 *   1. Reconstruct sign vector as +/- scale in rotated space
 *      (scale = sqrt(2/pi) / sqrt(dim), the expected absolute value
 *       of a half-normal distribution after RHT)
 *   2. Apply inverse RHT
 *   3. Scale by original L2 norm
 *
 * Note: This is a rough point-wise reconstruction. The real value
 * of 1-bit quantization is in Hamming attention (below).
 * ============================================================ */

void dequantize_row_tq_kv_1b(const block_tq_kv_1b * x, float * y, int64_t k) {
    const int block_size = TQ_KV_1B_BLOCK_SIZE;
    const int64_t num_blocks = k / block_size;

    for (int64_t b = 0; b < num_blocks; b++) {
        const block_tq_kv_1b * block = &x[b];
        float * dst = y + b * block_size;

        float norm = tq_fp16_to_fp32(block->norm);
        uint32_t seed = block->rht_seed;

        /* Reconstruct sign vector in rotated space.
         * After RHT, coordinates are ~N(0, 1/sqrt(dim)).
         * Expected |x| for half-normal = sqrt(2/pi) * sigma = sqrt(2/pi) / sqrt(dim). */
        float scale = sqrtf(2.0f / TQ_PI) / sqrtf((float)block_size);
        float rotated[TQ_KV_1B_BLOCK_SIZE];
        for (int i = 0; i < block_size; i++) {
            int bit = (block->signs[i / 8] >> (i % 8)) & 1;
            rotated[i] = bit ? scale : -scale;
        }

        /* Inverse RHT */
        tq_rht_inverse(rotated, block_size, seed);

        /* Scale by original norm */
        for (int i = 0; i < block_size; i++) {
            dst[i] = rotated[i] * norm;
        }
    }
}

/* ============================================================
 * Attention: XOR + popcount Hamming distance
 *
 * Ultra-fast attention using bitwise operations:
 *   1. RHT(query) computed ONCE (all keys share the same seed)
 *   2. Extract query sign bits ONCE
 *   3. Per key: XOR + popcount -> Hamming distance -> score
 *
 * Inner product estimator (from QJL/quant.cpp theory):
 *   <q, k> ~ q_norm * k_norm * sqrt(pi/2) / dim * (2*agree - dim)
 *
 * where agree = dim - hamming_distance(q_signs, k_signs).
 *
 * Theoretical cosine similarity: 2/pi ~ 0.637 for random vectors.
 * In practice, attention patterns are preserved because relative
 * ordering of scores is maintained (important tokens stay important).
 * ============================================================ */

/* ============================================================
 * VecDot: TQ_KV_1B (KV cache) dot F32 (query)
 *
 * This is a dequantize-based vec_dot for use in attention.
 * TQ_KV_1B is designed for KV cache compression, where:
 * - Keys/Values are quantized to 1-bit (TQ_KV_1B)
 * - Query is in FP32
 * - We need dot product between quantized key and query
 *
 * Since TQ_KV_1B -> F32 dequantization is very fast (O(n log n) RHT),
 * we dequantize the key and do a simple F32 dot product.
 *
 * Note: For production use, consider the XOR+popcount Hamming
 * attention (tq_kv_1b_attention) which is O(1) per key.
 * ============================================================ */

void ggml_vec_dot_tq_kv_1b_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                const void * GGML_RESTRICT vx, size_t bx,
                                const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = TQ_KV_1B_BLOCK_SIZE;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq_kv_1b * GGML_RESTRICT x = vx;  /* TQ_KV_1B (quantized key) */
    const float * GGML_RESTRICT y = vy;           /* F32 (query) */

    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const block_tq_kv_1b * block = &x[i];
        const float * query_block = y + i * qk;

        float norm = tq_fp16_to_fp32(block->norm);
        uint32_t seed = block->rht_seed;

        /* Reconstruct sign vector in rotated space */
        float scale = sqrtf(2.0f / TQ_PI) / sqrtf((float)qk);
        float rotated[TQ_KV_1B_BLOCK_SIZE];
        for (int j = 0; j < qk; j++) {
            int bit = (block->signs[j / 8] >> (j % 8)) & 1;
            rotated[j] = bit ? scale : -scale;
        }

        /* Inverse RHT */
        tq_rht_inverse(rotated, qk, seed);

        /* Scale by norm and compute dot product with query */
        for (int j = 0; j < qk; j++) {
            sumf += rotated[j] * norm * query_block[j];
        }
    }

    *s = sumf;
}

void tq_kv_1b_attention(const float * query, const block_tq_kv_1b * kv_cache,
                         float * scores, int seq_len, int head_dim) {
    const int dim = (head_dim <= TQ_KV_1B_BLOCK_SIZE) ? head_dim : TQ_KV_1B_BLOCK_SIZE;

    float scale_factor = sqrtf(TQ_PI_2) / (float)dim;

    /* Step 1: RHT(query) computed ONCE.
     * Since all keys use TQ_DEFAULT_SEED, a single rotation suffices.
     * RHT is orthogonal: <q, Pi^T * k_rot> = <Pi*q, k_rot>. */
    float q_rot[TQ_KV_1B_BLOCK_SIZE];
    memcpy(q_rot, query, (size_t)dim * sizeof(float));
    {
        int i;
        for (i = dim; i < TQ_KV_1B_BLOCK_SIZE; i++) q_rot[i] = 0.0f;
    }
    tq_rht_forward(q_rot, dim, TQ_DEFAULT_SEED);

    /* Step 2: Compute query L2 norm */
    float q_norm_sq = 0.0f;
    {
        int i;
        for (i = 0; i < dim; i++) {
            q_norm_sq += query[i] * query[i];
        }
    }
    float q_norm = sqrtf(q_norm_sq);

    /* Step 3: Extract query sign bits */
    int sign_bytes = dim / 8;
    uint8_t q_signs[TQ_KV_1B_BLOCK_SIZE / 8];
    if (sign_bytes > 0) memset(q_signs, 0, (size_t)sign_bytes);
    {
        int i;
        for (i = 0; i < dim; i++) {
            if (q_rot[i] > 0.0f) {
                q_signs[i / 8] |= (uint8_t)(1 << (i % 8));
            }
        }
    }

    /* Step 4: Per-key Hamming attention */
    {
        int seq;
        for (seq = 0; seq < seq_len; seq++) {
            const block_tq_kv_1b * blk = &kv_cache[seq];
            float k_norm = tq_fp16_to_fp32(blk->norm);

            /* XOR + popcount -> Hamming distance */
            int hamming = 0;
            {
                int b;
                for (b = 0; b < sign_bytes; b++) {
                    uint8_t xor_byte = q_signs[b] ^ blk->signs[b];
                    hamming += tq_popcount8(xor_byte);
                }
            }

            int agree = dim - hamming;
            float score = q_norm * k_norm * scale_factor * (float)(2 * agree - dim);
            scores[seq] = score;
        }
    }
}

/* ============================================================
 * Uniform 4-bit KV cache quantization (dequantize-based vec_dot)
 *
 * Simple min-max uniform scalar quantization:
 *   1. Find min/max in block
 *   2. Compute scale = (max - min) / 15 (16 levels for 4-bit)
 *   3. Map each value to nearest level: q = round((x - min) / scale)
 *   4. Pack 2 values per byte (LSB-first)
 *
 * Dequantization: x = min + (q + 0.5) * scale
 *
 * This is the same algorithm as quant.cpp's TQ_TYPE_UNIFORM_4B.
 * ============================================================ */

void quantize_row_tq_kv_4b_uniform_ref(const float * x, block_tq_kv_4b_uniform * y, int64_t k) {
    const int block_size = TQ_KV_4B_UNIFORM_BLOCK_SIZE;
    const int64_t num_blocks = k / block_size;

    for (int64_t b = 0; b < num_blocks; b++) {
        const float * src = x + b * block_size;
        block_tq_kv_4b_uniform * block = &y[b];

        /* Step 1: Find min/max */
        float mn = 1e30f, mx = -1e30f;
        int i;
        for (i = 0; i < block_size; i++) {
            float v = src[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }

        /* Step 2: Compute scale and zero_point */
        float range = mx - mn;
        if (range < 1e-8f) range = 1e-8f;  /* Avoid division by zero */
        float scale = range / 15.0f;        /* 16 levels: 0-15 */

        block->scale      = tq_fp32_to_fp16(scale);
        block->zero_point = tq_fp32_to_fp16(mn);

        /* Step 3: Quantize and pack */
        memset(block->qs, 0, (size_t)(block_size / 2));
        for (i = 0; i < block_size; i++) {
            int q = (int)((src[i] - mn) / scale + 0.5f);  /* Round to nearest */
            if (q < 0)  q = 0;
            if (q > 15) q = 15;

            /* LSB-first packing: 2 values per byte */
            if (i % 2 == 0) {
                block->qs[i / 2] = (uint8_t)q;
            } else {
                block->qs[i / 2] |= (uint8_t)(q << 4);
            }
        }
    }
}

void dequantize_row_tq_kv_4b_uniform(const block_tq_kv_4b_uniform * x, float * y, int64_t k) {
    const int block_size = TQ_KV_4B_UNIFORM_BLOCK_SIZE;
    const int64_t num_blocks = k / block_size;

    for (int64_t b = 0; b < num_blocks; b++) {
        const block_tq_kv_4b_uniform * block = &x[b];
        float * dst = y + b * block_size;

        float scale = tq_fp16_to_fp32(block->scale);
        float mn    = tq_fp16_to_fp32(block->zero_point);

        int i;
        for (i = 0; i < block_size; i++) {
            /* Extract 4-bit value */
            uint8_t byte = block->qs[i / 2];
            int q = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);

            /* Reconstruct: x = min + (q + 0.5) * scale */
            dst[i] = mn + ((float)q + 0.5f) * scale;
        }
    }
}

/* ============================================================
 * VecDot: TQ_KV_4B_UNIFORM (KV cache) dot F32 (query)
 *
 * Dequantize-based implementation:
 *   1. Dequantize KV block to F32
 *   2. Compute dot product with F32 query
 * ============================================================ */

void ggml_vec_dot_tq_kv_4b_uniform_f32(int n, float * GGML_RESTRICT s, size_t bs,
                                        const void * GGML_RESTRICT vx, size_t bx,
                                        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = TQ_KV_4B_UNIFORM_BLOCK_SIZE;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq_kv_4b_uniform * GGML_RESTRICT x = vx;  /* TQ_KV_4B_UNIFORM */
    const float * GGML_RESTRICT y = vy;                   /* F32 (query) */

    float sumf = 0.0f;
    int i;

    for (i = 0; i < nb; i++) {
        const block_tq_kv_4b_uniform * block = &x[i];
        const float * query_block = y + i * qk;

        float scale = tq_fp16_to_fp32(block->scale);
        float mn    = tq_fp16_to_fp32(block->zero_point);

        /* Dequantize and compute dot product on-the-fly */
        int j;
        for (j = 0; j < qk; j++) {
            uint8_t byte = block->qs[j / 2];
            int q = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            float deq = mn + ((float)q + 0.5f) * scale;
            sumf += deq * query_block[j];
        }
    }

    *s = sumf;
}
