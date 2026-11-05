# r119 实机问题回归与 r120 服务层修复

2026-11-05，192.168.18.206，Flash Next 自研后端，2×910B4，256k。原版 r119 提交 43b7c706。r120 保留 r119 的 engine、6 个原生算子库、权重、量化与 Python 包版本，只修改 HTTP 服务和请求校验。

## 结论

原版 r119 存在可复现的服务问题，不能视为并发流式场景下已验收。三路并发流式请求触发死锁，需要恢复服务；Unicode 流式乱码、结束原因误报和参数校验缺失也已复现。它们对应的代码在 r116 已存在；r119 对 server.py 只新增两个 health 开关字段。此前的单请求性能验收没有覆盖并发场景。

修复后真实模型 66 个检查项全部通过。底层推理未发现本轮样例可见的精度退化，不能把这些有限样例扩展为全面质量保证。

## 原版复现与修复

| 问题 | 原版实测 | 修复方式 |
|---|---|---|
| 并发流式死锁 | 3 路请求后 NPU 空闲、主线程 futex 等待，health 连续超时 | 已加载引擎无锁读取；独立初始化锁；异步请求排队，模型锁仅在线程内持有 |
| 特殊 Unicode 乱码 | 非流式 `𠮷🧪🚀 café Ελληνικά`；流式 `���������� café Ελληνικά` | DecodeStream 跨 token 缓冲 UTF-8；长度截断的残余后缀与完整 tokenizer.decode 对齐 |
| 流式结束原因误报 | max_tokens=1 非流式 length、流式 stop | 根据最后 token 是否为 EOS 返回 stop / length |
| 非法参数 500 或静默忽略 | 下表所列 | JSON、消息、长度、温度、n、model 校验，返回 400 / 404 |

| 测例 | 原版 HTTP | 修复后 HTTP |
|---|---:|---:|
| invalid_json | 500 | 400 |
| nonobject_body | 500 | 400 |
| missing_messages | 500 | 400 |
| bad_max_tokens | 500 | 400 |
| zero_max_tokens | 200 | 400 |
| bad_temperature | 500 | 400 |
| wrong_model | 200 | 404 |
| unsupported_n | 200 | 400 |

max_tokens=0 不再回退为 128。请求中的 model 若指定，必须匹配 /v1/models 的 ID，不再静默使用另一模型；省略 model 的旧调用方式继续支持。后端仍是 batch=1，n=2 明确拒绝。

异步排队也避免大量等待流耗尽默认 AnyIO 线程池；CPU 模拟实测 45 路请求排队通过。真实模型实测的是 3 路，生成安全串行，未宣称并行吞吐提高。诊断 loss/selfcheck 的计算与锁等待移到线程，文本准备也不阻塞事件循环。共享 tokenizer 操作串行化，单请求的统计在释放模型锁前保存。

## 实际回归覆盖

| 项目 | 结果 |
|---|---|
| 固定基础选择题 | 24/24 |
| 基础代码生成 | 8/8，29 个小输入功能断言 |
| 流式与非流式对照 | emoji、特殊汉字、中文、英文、JSON 全部一致 |
| 非法请求 | 10/10 按预期拒绝 |
| 会话状态 | A→A→B→A、立即重复缓存通过；30 个独立新会话通过 |
| 真实并发 | 3 路流式均完成，各自标识无串扰，健康检查零超时 |
| 流主动断开 | 后续请求正常完成 |
| 8k / 256k | 三处检索均 3/3；重复新增 prefill=0；追问正确且只计算少量新 token |
| NLL | 200-token 冻结文本回归通过 |
| CPU 接口单测 | 25 个测试通过，包括全部 Unicode token 截断位置、45 路排队、loss 与 stream 同时执行 |

历史 LiveCodeBench 100 题没有重跑。8 个代码题是基础功能样例，不是那套 100 题的复测。首轮沙箱缺少 set.add 与 str.isascii 的白名单，误拒绝了两个正确答案；保留原始答案和原始记录，加入安全方法后直接复判通过，没有重新采样答案。

真实三路生成期间 health 最长 3.47 ms，49 次采样零超时。流断开后新请求 0.681 秒完成。NLL 请求耗时 9.893 秒，期间 health 最长 3.48 ms、零超时。

## 性能与精度

| 实测 | TTFT | 输出 TPOT | 输出速度 |
|---|---:|---:|---:|
| 修复后 256k，全新输入 262016 / 输出 128 | 87.965 s | 47.09 ms | 21.24 tok/s |
| 修复后 8k，同口径三次中位数 | 5.300 s | 45.70 ms | 21.88 tok/s |

NLL=2.165267，与 r119 相同；冻结参考 2.164792±0.01。256k 前次 r119 生产值为 86.703 秒 / 21.29 tok/s；本次带健康采样的单次长测存在小幅波动，未观察到明显输出速度退化。8k 另以不带健康轮询的三次测量列示，首个综合长测不与历史三次中位数直接比较。

## 复现与回退

CPU 测试使用同一模型 tokenizer 和确定性 fake engine，不加载模型权重；需要测试环境可导入 httpx。本机通过独立 testdeps 引用容器全局的 httpx/httpcore，没有给生产 venv 安装新包。

```bash
PYTHONPATH="$PWD:$TEST_DEPS" /root/ascend-dev/venv/bin/python tests/r120_http.py
FNX_REGRESSION_DIR=/models/fnx-diagnostics/new-http-check   /root/ascend-dev/venv/bin/python tools/r120_http_regression.py
```

真实回归脚本会主动发送非法请求、并发请求、断开连接和长文请求，应安排测试窗口。结果保存在指定目录；API 提供的 model ID 以 /v1/models 为准。health 增加 service_revision=r120。

本机回退：bash /models/fnx-diagnostics/r120-service-20261105/rollback.sh，恢复原 r119 server.py；原版并发死锁也会一并恢复，因此该回退只用于诊断。r119 原始证据保存在 /models/fnx-diagnostics/r119-regression-20261105，修复和复测记录在 /models/fnx-diagnostics/r120-service-20261105。

增量解码使用本机已安装 tokenizers 0.23.3 的 [DecodeStream](https://huggingface.co/docs/tokenizers/api/decoders)。Git 保存复现数据与测试，不包含其他用户请求内容的完整服务日志。
