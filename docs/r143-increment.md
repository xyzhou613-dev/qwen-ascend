# Flash Next r143 增量升级

本包适用于已完成 r142 部署并通过环境自检的 2×910B4 主机。它只增加索引打分后处理融合，保留 r142 首块 2048 和其他分块策略。更早版本先按既有文档升级至 r142。

验证矩阵：CANN 9.1.0、驱动 26.1.1、torch 2.10.0、torch_npu 2.10.0.post4。其他矩阵先自检、重编译，并重新做精度和完整请求验收。

上传压缩包与 SHA256 文件到同一目录，先预检查：

```bash
sha256sum -c FlashNext-r143-increment.tgz.sha256
FNX_R143_STAGE=$(mktemp -d /root/fnx-r143-stage.XXXXXX)
tar -xzf FlashNext-r143-increment.tgz -C "$FNX_R143_STAGE"
python3 "$FNX_R143_STAGE/r143-enable.py" /root/ascend-dev/p275.sh --check
```

暂停外部自动恢复任务，停止模型、备份后安装。共享库必须在使用它的主进程和 worker 停止后替换。

```bash
/root/ascend-dev/modelctl.sh stop
FNX_R143_BACKUP="/root/ascend-dev/backup-before-r143-$(date +%Y%m%d-%H%M%S).tgz"
tar -czf "$FNX_R143_BACKUP" -C /root/ascend-dev fnx/fnx p275.sh
echo "回退备份：$FNX_R143_BACKUP"
tar -xzf FlashNext-r143-increment.tgz -C /root/ascend-dev
python3 /root/ascend-dev/r143-enable.py /root/ascend-dev/p275.sh
/root/ascend-dev/modelctl.sh start flash-next
```

补丁只新增 `FNX_INDEX_POST`，保留端口、卡号、容器名和 r142 已有配置。无法识别脚本时不写入。参数化默认值仍允许环境覆盖，应确认调用方没有设置 `FNX_INDEX_POST=0`。

源码默认关闭，通过 `FNX_INDEX_POST=1` 启用。生产固定使用 FP32 Div；乘倒数的 mode0 仅用于被否决的校准对照，不得启用。新核固定索引头维度 128、四 token 池化，其他模型语义需重新适配。

206 主机检查如下；其他主机使用原端口。启动包含多种计算图捕获，需要数分钟。

```bash
curl -fs http://127.0.0.1:9002/health
```

应包含 `service_revision=r143`、`index_post=true`、`ctx=262144`、`worker_alive=true`，并保留 `moe_epilogue=true`、`post_native=true`、`expert_gather=true`、`decode_down_gmm=false`。

升级后验证三类质量：原有 decode NLL、真实分块 prefill NLL、完整长上下文请求。`/fnx/loss` 只覆盖 decode，不能替代 prefill 评分。自然文档评分脚本及冻结 token IDs 位于仓库 `benchmarks/r143-index-post/`；避免在现有两卡模型旁再装载完整 Engine。

仅停用本轮：`FNX_INDEX_POST=0` 重启。完整回退：先停模型，解压记录的备份回 `/root/ascend-dev`，再通过既有入口启动。

本机编译器位于 Flash Next 容器。可先将源码复制进去，编译到临时目录，验证后停模型再安装：

```bash
FNX_R143_BUILD=$(docker exec bole-model-Flash mktemp -d /tmp/fnx-r143-build.XXXXXX)
docker cp /root/qwen-ascend/kernels/index_post "bole-model-Flash:$FNX_R143_BUILD/index_post"
docker exec -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest bole-model-Flash bash "$FNX_R143_BUILD/index_post/build.sh" "$FNX_R143_BUILD/libfnx_index_post.so"
```

增量包仅包含四个运行文件：engine.py、server.py、index_post.py、libfnx_index_post.so，以及参数补丁脚本。不含权重，不替换已有 gather 或其他共享库。

后续模型切换入口保留：

```bash
/root/ascend-dev/modelctl.sh start 27b
/root/ascend-dev/modelctl.sh start flash-next
```

本轮未重新验收 27B 来回切换，也未重跑历史 100 题。详细数据见 `FlashNext-r143-优化实测-20261112.md`。
