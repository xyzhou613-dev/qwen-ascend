# r165：FNX 工具调用协议修复与上线验收（2026-11-14）

**已上线 192.168.18.206:9002。推理版本保持 r152，HTTP 工具协议为 r165。** `/health.tool_protocol_revision` 返回 `r165`。本轮未修改模型权重、采样计算或推理内核。

此前核查发现当前实例仍把结构化参数当字符串、忽略 tool_choice、缺少流式 index/usage；另有输出预算双字段优先级和 null 回退问题。用户随后明确要求修复。本轮修复对象是当前206实例，不涉及文档中另一台16.11实例。

## 修复内容

| 项目 | 上线行为 |
|---|---|
| 参数类型 | 按对应工具的schema恢复整数、数字、布尔、数组、对象和null；字符串声明保留数字/true/null等文本 |
| Python字面量 | JSON优先，兼容True/False/None及单引号对象；不执行表达式 |
| schema引用和联合类型 | 支持本地$ref、基本anyOf/oneOf与嵌套类型检查；避免把bool当integer |
| tool_choice | auto保留模型选择；none移除工具声明并按文本输出；required/指定函数使用Qwen续写前缀，单工具保持完整MCP名称；非法选择和未声明函数返回400 |
| 流式调用 | delta.tool_calls补充连续index；include_usage=true时输出标准choices=[]的usage块，然后DONE |
| 输出预算 | max_completion_tokens非null时优先；否则回退max_tokens；两者均无值才默认128，仍检查正整数和上下文容量 |
| 已有行为回归 | 保留DecodeStream中文/emoji解码、length/stop/tool_calls结束原因、非流式usage及既有fnx_stats |

外层function.arguments仍为标准JSON字符串；其内部字段现在保留正确类型。实际返回内容示例：

```json
{
  "count": 3,
  "enabled": true,
  "tags": [
    "甲",
    "乙"
  ],
  "options": {
    "mode": "fast"
  },
  "label": "123"
}
```

未声明schema的旧消息式XML用法继续保留字符串行为。无法按声明类型解析、未知函数、参数重复或不完整XML保留在文本中，不伪装成有效调用。强制工具选择使用续写前缀，仍需给足生成预算；本轮没有实现完整的JSON Schema约束解码。

## 验证结果

39项纯协议测试、28组实际请求处理函数测试全部通过。处理函数测试使用真实tokenizer和确定性生成器，禁止构造模型，NPU未初始化。另对修复前捕获的3份真实模型参数重放，数组/对象/数字/布尔转换及六项todos恢复均通过。

| 真实接口验收 | 结果 |
|---|---|
| 非流式、流式结构化参数 | count=int、enabled=bool、tags=list、options=dict、label=str |
| 流式index/usage | index=0，completion_tokens=84，总计数=prompt+completion |
| 单工具required | 完整mcp__audit__echo，中文及😀完整 |
| 多工具required、指定函数 | 返回已注册函数；指定函数为准确完整名称 |
| tool_choice=none | 文本响应，无结构化调用 |
| 非法选择、不存在的函数 | 两项均为预期HTTP400 |
| 现代字段、双字段、旧字段null | 三项长调用均生成162 token，返回六项todos数组 |
| 32 token截断 | finish_reason=length |

上述共12项，全部通过。再补两项真实接口检查：按index拼装流式调用并回传带数组/对象的工具结果，下一轮正确返回“协议回传成功”；enable_thinking=true下的required完整MCP回声调用也通过。没有执行返回的外部工具，未运行真实C SDK程序；因此不把这些接口结果表述为C SDK端到端验收。

普通生成回归：新鲜8k输入、128输出与原r152基线回答完全一致，单次TPOT 33.09ms。该项用于回归，不作为新的性能提速结论。

## 发布、备份与恢复

本轮在空闲确认后，从 17:17:00 启动维护，17:22:31 验收完成，总耗时 331.1 秒（约5.52分钟）。根据本次修复指令和原任务必要短维护授权执行；20分钟为本轮自设保护上限，不沿用已结束的r164性能窗口。独立回滚定时为17:27:00，第二守护晚15秒触发。

只替换server.py、request_validation.py并新增tool_protocol.py。43个其他推理源文件、p275启动脚本和原容器身份均保持。先在临时目录执行了完整安装/回退字节校验；正式部署通过后没有触发回滚。原文件备份保存在诊断目录的backup/。

正常开机恢复保持启用；维护期间额外保护未验收源码在新boot前回退。验收后已清理该临时boot hook、回滚timer、guard、manager临时drop-in、lease和捕获的私有环境。manager、单watchdog及双进程worker正常。回滚程序刻意拒绝撤回已经accepted的发布；未来如需人工回退，应新建维护记录并使用备份，不重置本次窗口。

源码与证据：fnx/tool_protocol.py、fnx/server.py、fnx/request_validation.py；tests/t_tool_protocol.py、tests/t_server_tool_protocol.py；benchmarks/r165-tool-protocol保存原始请求/响应、测试和恢复证明，experiments/r165-tool-protocol保存本轮控制脚本。
