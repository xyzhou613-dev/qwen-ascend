/* 第 39 轮：MIX kernel 可行性验证 —— host 侧（tiling + shape/dtype 推断）
 *
 * 关键点：tiling 用**框架的 MultiCoreMatmulTiling** 算（第 27 轮已单独验证过，
 * 框架外能拿到 TCubeTiling），然后整块交给 kernel；SetBlockDim 用 tiling 给的核数。
 * 工作区给 16 MiB：Matmul 的 KFC（Cube/Vector 协同）需要 workspace 放通信区。
 */
#include "../op_kernel/cube_probe_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matrix/bmm_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CubeProbeTilingData* tiling = context->GetTilingData<CubeProbeTilingData>();
    const gert::StorageShape* xs = context->GetInputShape(0);
    const gert::StorageShape* ws = context->GetInputShape(1);
    if (xs == nullptr || ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t M = xs->GetStorageShape().GetDim(0);
    int64_t K = xs->GetStorageShape().GetDim(1);
    int64_t N = ws->GetStorageShape().GetDim(1);
    tiling->m = static_cast<uint32_t>(M);
    tiling->n = static_cast<uint32_t>(N);
    tiling->k = static_cast<uint32_t>(K);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling api(platform);
    int32_t aicNum = static_cast<int32_t>(platform.GetCoreNumAic());
    if (aicNum <= 1) { aicNum = 20; }
    printf("[tiling] coreNumAic=%d usedCoreNum=%d", (int)aicNum, (int)aicNum);
    api.SetDim(aicNum);
    api.SetShape(M, N, K);
    api.SetOrgShape(M, N, K);
    api.SetBias(false);
    api.SetBufferSpace(-1, -1, -1);
    if (api.GetTiling(tiling->mm) == -1) {
        return ge::GRAPH_FAILED;
    }

    size_t* curWs = context->GetWorkspaceSizes(1);
    curWs[0] = 16 * 1024 * 1024;
    context->SetBlockDim(tiling->mm.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    const gert::Shape* w = context->GetInputShape(1);
    gert::Shape* y = context->GetOutputShape(0);
    y->SetDimNum(0);
    y->AppendDim(x->GetDim(0));       // M
    y->AppendDim(w->GetDim(1));       // N
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class CubeProbe : public OpDef {
public:
    explicit CubeProbe(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(CubeProbe);
}  // namespace ops
