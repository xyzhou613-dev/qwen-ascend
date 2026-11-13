# r160 UB LUT down（组件更慢，未部署）

实际结果见../../docs/r160-ub-lut-down.md。代码在本机CANN9.1编译，53样本×512专家数值与r158逐位相同，动态graph通过；独占down+epilogue比生产更慢，未测真实层/全模型。

本机固定路径脚本，不是新环境一键启动器。probe.py默认仅做在线数值验证，并在发现生产请求进展时退出，不构造Engine；--timing只允许独占状态及有效lease/截止。单个真实down权重、1/32 allocator，禁止在正式模型占卡时假称独占结果。

start_micro.py只沿用当时明确授权的r159原绝对截止，不能重置或挪用已结束的窗口；后续复现需要新的有效范围及独立目录。原入口、manager、watchdog由独立rescue恢复；绝对timer和独立guardian保留至恢复完成，随后finalize清理并核验。

恢复后formal_smoke只执行http_bench.py的SSE前缀函数，未执行其中整套候选性能请求。不要把恢复烟测称为新候选TPOT。
