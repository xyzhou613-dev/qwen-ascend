/* 第 39 轮：MIX kernel 可行性验证 —— aclnn 调用程序
 *
 * 用法：probe_host <M> <N> <K> [iters]
 *   建 x[M,K] fp16 / w[K,N] fp16（小随机数），算 y=x@w，与 CPU fp32 参考比 rel；
 *   然后循环计时，报 ms/次 与等效 TFLOPS。
 *
 * 关键：这条调用路径走的是**框架的 aclnn**（不是第 26/27 轮那条直调 kernel 的路），
 * 所以 tiling / L1 / L0 / 流水都由框架负责 —— 这正是"自定义 op 能不能跑 Cube"要验的东西。
 */
#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" aclnnStatus aclnnCubeProbeGetWorkspaceSize(const aclTensor* x, const aclTensor* w,
                                                      const aclTensor* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);
extern "C" aclnnStatus aclnnCubeProbe(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor, aclrtStream stream);

#define CK(expr, msg)                                        \
    do {                                                     \
        auto _r = (expr);                                    \
        if (_r != 0) {                                       \
            printf("[FAIL] %s rc=%d\n", msg, (int)_r);       \
            return 1;                                        \
        }                                                    \
    } while (0)

static uint16_t f2h(float f)
{
    // 极简 float -> 半精度（够用于随机小数值）
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (man >> 13));
}

static float h2f(uint16_t h)
{
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        out = sign;
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

int main(int argc, char** argv)
{
    int M = (argc > 1) ? atoi(argv[1]) : 16;
    int N = (argc > 2) ? atoi(argv[2]) : 1280;
    int K = (argc > 3) ? atoi(argv[3]) : 2560;
    int iters = (argc > 4) ? atoi(argv[4]) : 50;
    printf("=== CubeProbe: M=%d N=%d K=%d iters=%d ===\n", M, N, K, iters);

    CK(aclInit(nullptr), "aclInit");
    CK(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CK(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::vector<uint16_t> hx((size_t)M * K), hw((size_t)K * N), hy((size_t)M * N, 0);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    for (auto& v : hx) v = f2h(dist(rng));
    for (auto& v : hw) v = f2h(dist(rng));

    void *dx = nullptr, *dw = nullptr, *dy = nullptr;
    CK(aclrtMalloc(&dx, hx.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST), "malloc dx");
    CK(aclrtMalloc(&dw, hw.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST), "malloc dw");
    CK(aclrtMalloc(&dy, hy.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST), "malloc dy");
    CK(aclrtMemcpy(dx, hx.size() * 2, hx.data(), hx.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "cp x");
    CK(aclrtMemcpy(dw, hw.size() * 2, hw.data(), hw.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "cp w");
    CK(aclrtMemset(dy, hy.size() * 2, 0, hy.size() * 2), "zero y");

    int64_t xs[2] = {M, K}, ws[2] = {K, N}, ys[2] = {M, N};
    int64_t st[2] = {K, 1}, stw[2] = {N, 1}, sty[2] = {N, 1};
    aclTensor* tx = aclCreateTensor(xs, 2, ACL_FLOAT16, st, 0, ACL_FORMAT_ND, xs, 2, dx);
    aclTensor* tw = aclCreateTensor(ws, 2, ACL_FLOAT16, stw, 0, ACL_FORMAT_ND, ws, 2, dw);
    aclTensor* ty = aclCreateTensor(ys, 2, ACL_FLOAT16, sty, 0, ACL_FORMAT_ND, ys, 2, dy);
    if (!tx || !tw || !ty) { printf("[FAIL] aclCreateTensor\n"); return 1; }

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus r = aclnnCubeProbeGetWorkspaceSize(tx, tw, ty, &wsSize, &exec);
    printf("GetWorkspaceSize rc=%d wsSize=%llu\n", (int)r, (unsigned long long)wsSize);
    if (r != 0) { printf("[FAIL] GetWorkspaceSize\n"); return 1; }
    void* wsPtr = nullptr;
    if (wsSize > 0) {
        if (aclrtMalloc(&wsPtr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
            printf("[FAIL] malloc workspace\n"); return 1;
        }
    }
    r = aclnnCubeProbe(wsPtr, wsSize, exec, stream);
    printf("aclnnCubeProbe rc=%d\n", (int)r);
    if (r != 0) { printf("[FAIL] 执行失败\n"); return 1; }
    CK(aclrtSynchronizeStream(stream), "sync");
    CK(aclrtMemcpy(hy.data(), hy.size() * 2, dy, hy.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST), "cp y");

    // ---- 与 CPU fp32 参考比 ----
    double num = 0.0, den = 0.0, mx = 0.0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)h2f(hx[(size_t)i * K + k]) * (double)h2f(hw[(size_t)k * N + j]);
            double got = (double)h2f(hy[(size_t)i * N + j]);
            num += std::fabs(got - acc);
            den += std::fabs(acc);
            if (std::fabs(got - acc) > mx) mx = std::fabs(got - acc);
        }
    }
    double rel = den > 0 ? num / den : 0.0;
    printf("数值：rel_mean=%.3e  max_abs=%.3e  y[0]=%.6f  -> %s\n", rel, mx,
           (double)h2f(hy[0]), (M * N <= 64 && (den == 0.0)) ? "全零(说明 kernel 没写)" : (rel < 3e-2 ? "PASS" : "FAIL"));

    // ---- 计时 ----
    for (int t = 0; t < 5; ++t) {
        aclnnCubeProbeGetWorkspaceSize(tx, tw, ty, &wsSize, &exec);
        aclnnCubeProbe(wsPtr, wsSize, exec, stream);
    }
    CK(aclrtSynchronizeStream(stream), "warm sync");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int t = 0; t < iters; ++t) {
        aclnnCubeProbeGetWorkspaceSize(tx, tw, ty, &wsSize, &exec);
        aclnnCubeProbe(wsPtr, wsSize, exec, stream);
    }
    CK(aclrtSynchronizeStream(stream), "time sync");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = ((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6) / iters;
    double flops = 2.0 * M * N * K;
    printf("时间：%.3f ms/次  等效 %.1f TFLOPS  (占峰值 241T 的 %.1f%%)\n",
           ms, flops / (ms * 1e-3) / 1e12, flops / (ms * 1e-3) / 1e12 / 241.0 * 100.0);

    aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(ty);
    aclrtFree(dx); aclrtFree(dw); aclrtFree(dy);
    if (wsPtr) aclrtFree(wsPtr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
