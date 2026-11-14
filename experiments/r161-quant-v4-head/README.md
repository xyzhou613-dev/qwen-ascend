# r161 在线组件实验，未部署

结果见../../docs/r161-quant-v4-head.md。脚本依赖现场r154私有校准与正式模型权重，路径固定；不包含这些私有输入。prepare.py从冻结r154 runtime构建副本，small_delta沿用r153。禁止在正式占卡时启动第二份完整Engine。

所有结果exclusive:false；单层代理、head组件和热状态恢复均不能当作HTTP TPOT。packed conv已拒绝；QuantHead仅为需完整质量验收的候选。
