# r159 BF16 down 实测：组件门槛未通过

实际结果见../../docs/r159-bf16-down-validation.md与相邻benchmarks目录。生产继续r152。

本机本轮执行脚本，路径固定，不是新环境一键启动器。首次start_window.py的systemd append日志打开失败；实际通过launch-retry.json/measurement-retry.json所记的进程内重定向启动。component_probe.py在初次审计登记错误后修正；原始版本与初次manifest在initial-component-attempt保留。

实际执行：prepare/preflight、单权重组件比较和select_component门槛、独立rescue、finalize_completed_run。controller_retry.sh计划的整层、NLL/prefill/HTTP后续阶段没有执行。planned-scope.md是起初计划，不代表已测覆盖。

组件门槛失败后未降低标准或进入整模型。不得在正式Engine占卡时运行独占探针，不得重置window-start沿用已经结束的授权。复现需新的明确时间窗口与独立目录，并保留原入口恢复保护。

依赖r157原私有runtime与r158固定二进制；prepare.py记录实际生成方式，candidate-kmajor-down.py与candidate-audit.py是本轮实际集成模块，runtime-code-manifest.json可核查。编译器/模型环境沿用前轮。完整模型和任务输入只在服务器，未纳入公开包。
