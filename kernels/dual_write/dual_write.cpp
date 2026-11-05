#include "kernel_operator.h"
#include "acl/acl.h"

using namespace AscendC;

namespace {
constexpr int Width = 256;
constexpr int Heads = 2;
constexpr int Row = Heads * Width;
constexpr int Tile = 16;
constexpr int ElementsPerBlock = 16;

__aicore__ inline DataCopyParams bursts(int rows, int length, int src_stride, int dst_stride)
{
    return {static_cast<uint16_t>(rows),
            static_cast<uint16_t>(length / ElementsPerBlock),
            static_cast<uint16_t>((src_stride - length) / ElementsPerBlock),
            static_cast<uint16_t>((dst_stride - length) / ElementsPerBlock)};
}
}

extern "C" __global__ __aicore__ void write_dual_kv(
    GM_ADDR key, GM_ADDR value, GM_ADDR positions, GM_ADDR original_k,
    GM_ADDR original_v, GM_ADDR latent, GM_ADDR rope,
    int tokens, int capacity, int key_stride, int value_stride)
{
    GlobalTensor<half> k, v, ok, ov, packed, rotary;
    GlobalTensor<int64_t> pos;
    k.SetGlobalBuffer((__gm__ half*)key);
    v.SetGlobalBuffer((__gm__ half*)value);
    ok.SetGlobalBuffer((__gm__ half*)original_k);
    ov.SetGlobalBuffer((__gm__ half*)original_v);
    packed.SetGlobalBuffer((__gm__ half*)latent);
    rotary.SetGlobalBuffer((__gm__ half*)rope);
    pos.SetGlobalBuffer((__gm__ int64_t*)positions);
    const int64_t start = pos.GetValue(0);
    if (start < 0 || start > capacity - tokens) return;

    TPipe pipe;
    TBuf<TPosition::VECCALC> kb, vb, zb;
    pipe.InitBuffer(kb, Tile * Row * sizeof(half));
    pipe.InitBuffer(vb, Tile * Row * sizeof(half));
    pipe.InitBuffer(zb, Tile * 64 * sizeof(half));
    auto keys = kb.Get<half>();
    auto values = vb.Get<half>();
    auto zeros = zb.Get<half>();
    Duplicate(zeros, static_cast<half>(0.0f), Tile * 64);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    const int tile = tokens < Tile * GetBlockNum() ? 1 : Tile;
    for (int row = GetBlockIdx() * tile; row < tokens; row += GetBlockNum() * tile) {
        const int count = tokens - row < tile ? tokens - row : tile;
        DataCopy(keys, k[static_cast<int64_t>(row) * key_stride], bursts(count, Row, key_stride, Row));
        DataCopy(values, v[static_cast<int64_t>(row) * value_stride], bursts(count, Row, value_stride, Row));
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);

        const int64_t offset = start + row;
        DataCopy(ok[offset * Row], keys, count * Row);
        DataCopy(ov[offset * Row], values, count * Row);
        for (int head = 0; head < Heads; ++head) {
            const int64_t packed_row = head * static_cast<int64_t>(capacity) + offset;
            const int col = head * Width;
            DataCopy(rotary[packed_row * 64], keys[col], bursts(count, 64, Row, 64));
            DataCopy(packed[packed_row * 512], keys[col + 64], bursts(count, 192, Row, 512));
            DataCopy(packed[packed_row * 512 + 192], values[col], bursts(count, 256, Row, 512));
            DataCopy(packed[packed_row * 512 + 448], zeros, bursts(count, 64, 64, 512));
        }
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}

extern "C" void fnx_write_dual_kv(
    void* key, void* value, void* positions, void* original_k, void* original_v,
    void* latent, void* rope, int tokens, int capacity, int key_stride,
    int value_stride, void* stream)
{
    const int blocks = tokens < 40 ? tokens : 40;
    write_dual_kv<<<blocks, nullptr, (aclrtStream)stream>>>(
        (GM_ADDR)key, (GM_ADDR)value, (GM_ADDR)positions, (GM_ADDR)original_k,
        (GM_ADDR)original_v, (GM_ADDR)latent, (GM_ADDR)rope,
        tokens, capacity, key_stride, value_stride);
}
