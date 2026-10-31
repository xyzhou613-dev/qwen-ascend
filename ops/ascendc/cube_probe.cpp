/* 第 39/40 轮：MIX kernel 可行性验证 —— kernel 侧（v3，多核手动偏移）
 *
 * v2 的现象（第 39 轮）：单核 PASS、20 核 rel 0.95 FAIL。
 * 第 40 轮把 block 拓扑打出来才看清：
 *     [AIV] blk=0/20  usedCoreNum=20  singleM=205  singleN=16   (M=1024,N=64)
 *   —— tiling 把 M 和 N **都切了**（20 = 5 个 M 块 × 4 个 N 块），而 kernel 里每个核
 *   都把结果写到偏移 0，于是只有最后一个核的 tile 留下来 → 其余全是过期数据。
 *   ⇒ 多核时**偏移必须由 kernel 自己加**（KFC client 不做这件事），尾块用 SetTail。
 *
 * 形状：y[M,N] = x[M,K] @ w[K,N]，全 fp16 ND。
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

    const uint32_t M = tilingData.m;
    const uint32_t N = tilingData.n;
    const uint32_t K = tilingData.k;
    const uint32_t coreNum = tilingData.mm.usedCoreNum;
    const uint32_t sM = tilingData.mm.singleCoreM;
    const uint32_t sN = tilingData.mm.singleCoreN;
    const uint32_t blk = static_cast<uint32_t>(GetBlockIdx());
    if (blk >= coreNum) {
        return;
    }

    // 核编号按 (mIdx, nIdx) 展开：本机实测 MultiCoreMatmulTiling 先铺 N 再进 M。
    const uint32_t nCoreNum = (sN != 0) ? ((N + sN - 1) / sN) : 1;
    const uint32_t mIdx = blk / nCoreNum;
    const uint32_t nIdx = blk % nCoreNum;
    const uint32_t m0 = mIdx * sM;
    const uint32_t n0 = nIdx * sN;
    if (m0 >= M || n0 >= N) {
        return;
    }
    const uint32_t curM = (m0 + sM <= M) ? sM : (M - m0);
    const uint32_t curN = (n0 + sN <= N) ? sN : (N - n0);

    GlobalTensor<half> gx;
    GlobalTensor<half> gw;
    GlobalTensor<half> gy;
    gx.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x) + (size_t)m0 * K, curM * K);
    gw.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w) + n0, K * N);
    gy.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y) + (size_t)m0 * N + n0, curM * N);

    mm.SetOrgShape(M, N, K);
    mm.SetTail(curM, curN, K);
    mm.SetTensorA(gx, false);
    mm.SetTensorB(gw, false);
    mm.IterateAll(gy, false);
    mm.End();
}
