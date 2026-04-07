#include "common.cuh"
#include "ggml-turbo-quant.h"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;
    const float d = x[ib].d;
    const float neg_d = -d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    const uint8_t bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const uint8_t bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = bit_0 ? d : neg_d;
    v.y = bit_1 ? d : neg_d;
}

static __device__ __forceinline__ void dequantize_q1_0_g128(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0_g128 * x = (const block_q1_0_g128 *) vx;
    const float d = x[ib].d;
    const float neg_d = -d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    const uint8_t bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const uint8_t bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = bit_0 ? d : neg_d;
    v.y = bit_1 ? d : neg_d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// TQ_KV_4B_UNIFORM: 4-bit uniform quantization with scale and zero_point
// Block size = 128, each byte holds 2 x 4-bit values (LSB-first)
// Reconstruct: x = zero_point + (q + 0.5) * scale
static __device__ __forceinline__ void dequantize_tq_kv_4b_uniform(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_tq_kv_4b_uniform * x = (const block_tq_kv_4b_uniform *) vx;

    // scale and zero_point are uint16_t storing FP16 bit patterns
    const float scale   = __half2float(*(const half*)&x[ib].scale);
    const float zero_pt = __half2float(*(const half*)&x[ib].zero_point);

    // For TQ_KV_4B_UNIFORM:
    // - qk=128, qr=1, so iqs = (i00 % 128) / 1
    // - But each byte holds 2 values: qs[i/2] gives the byte for value i
    // - iqs increments by 2 each iteration (from getrows kernel loop: i00 = 2*(...))
    const int byte_idx = iqs / 2;
    const uint8_t vui = x[ib].qs[byte_idx];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = zero_pt + (v.x + 0.5f) * scale;
    v.y = zero_pt + (v.y + 0.5f) * scale;
}
