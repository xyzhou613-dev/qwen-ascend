/* 第 39 轮：MIX kernel 可行性验证 —— kernel 侧（v2，GlobalTensor 版）
 *
 * v1 的编译错误是 "no matching member function for call to 'SetTensorA/B/IterateAll'"：
 * 高层 Matmul 的参数是 GlobalTensor，不是裸 __gm__ 指针。
 *
 * 形状：y[M,N] = x[M,K] @ w[K,N]，全 fp16 ND。
 * 这一版刻意不写解包、不碰 4-bit —— 先回答"自定义 op 能不能真的跑到 Cube"。
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "cube_probe_tiling.h"

using namespace AscendC;

using A_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using B_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using C_T = MatmulType<TPosition::GM, CubeFormat::ND, half>;

extern "C" __global__ __aicore__ void cube_probe(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CubeProbeTilingData);
    GET_TILING_DATA(tilingData, tiling);

    TPipe pipe;
    Matmul<A_T, B_T, C_T> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.mm);

    uint32_t M = tilingData.m;
    uint32_t N = tilingData.n;
    uint32_t K = tilingData.k;

    GlobalTensor<half> gx;
    GlobalTensor<half> gw;
    GlobalTensor<half> gy;
    gx.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), M * K);
    gw.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w), K * N);
    gy.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), M * N);

    mm.SetOrgShape(M, N, K);
    mm.SetTensorA(gx, false);
    mm.SetTensorB(gw, false);
    mm.IterateAll(gy, false);
    mm.End();
}
