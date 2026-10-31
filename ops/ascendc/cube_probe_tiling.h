/* 第 39 轮：MIX kernel 可行性验证 —— tiling 数据结构
 *
 * 为什么要看一眼这个文件：CANN 的 op-package 流程把"宿主算好的 tiling"当裸字节拷进
 * kernel，所以两侧必须**结构体布局一致**。这里直接内嵌框架的 TCubeTiling
 * （Matmul 高层 API 唯一的必需输入：M/N/Ka/Kb 分块、深度、核数、L1/L0 缓冲大小等）。
 */
#ifndef CUBE_PROBE_TILING_H
#define CUBE_PROBE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct CubeProbeTilingData {
    TCubeTiling mm;        // Matmul 高层 API 的 tiling
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

#endif // CUBE_PROBE_TILING_H
