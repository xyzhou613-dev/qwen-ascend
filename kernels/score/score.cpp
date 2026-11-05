#include "kernel_operator.h"
#include "acl/acl.h"

using namespace AscendC;
namespace {
constexpr int Tile = 2048;
constexpr float Negative = -1.0e30f;
}

extern "C" __global__ __aicore__ void score_groups(
    GM_ADDR source, GM_ADDR positions, GM_ADDR destination, int tokens, int groups)
{
    GlobalTensor<float> input, output;
    GlobalTensor<int64_t> pos;
    input.SetGlobalBuffer((__gm__ float*)source);
    pos.SetGlobalBuffer((__gm__ int64_t*)positions);
    output.SetGlobalBuffer((__gm__ float*)destination);
    TPipe pipe;
    TBuf<TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, 4 * Tile * sizeof(float));
    auto a = scratch.Get<float>();
    auto b = a[Tile];
    auto c = a[2 * Tile];
    auto d = a[3 * Tile];
    const int columns = groups / Tile;
    const int tasks = tokens * columns;
    for (int task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        const int row = task / columns;
        const int col = (task % columns) * Tile;
        const int64_t remain = (pos.GetValue(row) + 1) / 4 - col;
        const int valid = remain > Tile ? Tile : (remain > 0 ? remain : 0);
        const int load = (valid + 7) / 8 * 8;
        if (load) {
            const int64_t offset = static_cast<int64_t>(row) * 4 * groups + col;
            DataCopy(a, input[offset], load);
            DataCopy(b, input[offset + groups], load);
            DataCopy(c, input[offset + 2 * groups], load);
            DataCopy(d, input[offset + 3 * groups], load);
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            Relu(a, a, load);Relu(b, b, load);Relu(c, c, load);Relu(d, d, load);
            PipeBarrier<PIPE_V>();
            Add(a, a, b, load);PipeBarrier<PIPE_V>();
            Add(a, a, c, load);PipeBarrier<PIPE_V>();
            Add(a, a, d, load);PipeBarrier<PIPE_V>();
        }
        if (load < Tile) Duplicate(a[load], Negative, Tile - load);
        if (load != valid) {
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
            for (int col = valid; col < load; ++col) a.SetValue(col, Negative);
            SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
        }
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(output[static_cast<int64_t>(row) * groups + col], a, Tile);
        SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
    }
}

extern "C" void fnx_score_groups(void* input, void* positions, void* output,
                                 int tokens, int groups, void* stream)
{
    const int tasks = tokens * (groups / Tile);
    const int blocks = tasks < 40 ? tasks : 40;
    score_groups<<<blocks, nullptr, (aclrtStream)stream>>>(
        (GM_ADDR)input, (GM_ADDR)positions, (GM_ADDR)output, tokens, groups);
}
