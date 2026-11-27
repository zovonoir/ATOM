# Qwen3.8-Flash-Next 1P1D PD 分离（基于 qwen38-flash-next-20261102 镜像）

在已有的 `atom-sglang:qwen38-flash-next-20261102` 镜像中，把 ATOM 换成 `qwen38-pd-fix-v2` 分支、SGLang / aiter 换成
`qwen38-pd-fix` 分支，即可跑 SGLang + ATOM 的单节点 1P1D（P、D 各 TP2，MTP，P 端 radix，纯文本与多模态）。
不需要新镜像。

## 分支内容

| 组件 | 分支 | 基线 | 分支上的改动 |
|---|---|---|---|
| ATOM | `zovonoir/ATOM@qwen38-pd-fix-v2` | ROCm/ATOM `f3d0b30` | cherry-pick ROCm/ATOM#3521（gfx942 纯 TP MoE padding / 调优 dispatch、BF16 MTP 合并 shared expert）、#3887（PD 传输 Native QSA / PLE 状态）、#3632（prefill 序列并行通信走 SDMA）；`ATOM_FLASH_PD_ALLOW_RADIX=1` 允许 P 端开 radix；本目录脚本 |
| SGLang | `zovonoir/sglang@qwen38-pd-fix` | sgl-project/sglang `baa9bef`（v0.5.22） | ROCm pyproject；MTP 开启后思考过程死循环的修复（HIP 上按采样参数做 EAGLE 验证，不做全词表排序，即 v0.5.20 交付时 `1a9977fe` 补丁的移植）；PD 首步稀疏 draft proposal |
| aiter | `zovonoir/aiter@qwen38-pd-fix` | ROCm/aiter `fb3f506`（composable_kernel `f93c029`） | 无改动（已包含 ROCm/aiter#6576、#8148、#8149） |

另外需要 `flydsl==0.5.3`（镜像里是 0.4.1，新 aiter 要求 >= 0.5）和 `amd-mori-nightly==1.2.6.dev20261124`
（镜像里是 1.2.5，#3632 的 SDMA 需要 >= 1.2.6.dev20261116），均来自 PyPI。镜像里的 PyTorch / Triton / Mooncake / atomesh 不变。

- `EP_SIZE=1`（纯 TP）时 #3521 的路径才生效；默认 `EP_SIZE=2`，两种都已验证可运行。
- `ATOM_QWEN4_EXP_SP_SDMA=1`（`pd_env.sh` 默认开）只影响超过 1024 token 的 prefill，不影响 decode。

## 安装（容器内执行一次）

```bash
git clone --branch qwen38-pd-fix-v2 --single-branch https://github.com/zovonoir/ATOM.git /opt/qwen38-pd-fix/ATOM
bash /opt/qwen38-pd-fix/ATOM/recipes/qwen38_pd_fix/install.sh
```

`install.sh` 会：克隆 SGLang、aiter（含 composable_kernel 子模块）分支；安装 flydsl 0.5.3 和 amd-mori-nightly 1.2.6；
把 `/app/ATOM`、`/app/sglang`、`/app/aiter-test` 替换为分支代码（原目录保留为 `/app/*.orig-<时间戳>`）；
清掉 aiter 的 JIT 产物，首次启动时为 gfx942 重新编译（约 15 分钟，之后复用）。

无外网时：在别处把三个分支 clone 到同一目录（aiter 带 `--recurse-submodules`）并下载 flydsl、amd-mori-nightly 的 wheel，
拷进容器后执行 `SRC=<该目录> FLYDSL_WHEEL=<路径> MORI_WHEEL=<路径> bash install.sh`。

## 启动

模型默认在 `/model`（与交付报告一致），P 用 GPU 0,1，D 用 GPU 6,7。按需修改 `pd_env.sh`（模型路径、卡号、端口、CPU 绑定）。
三个终端分别执行（在本目录下）：

```bash
bash ./start_prefill.sh   # P：HTTP 7081，bootstrap 8998，radix 开
bash ./start_decode.sh    # D：HTTP 7082，radix 关
bash ./start_router.sh    # P、D 都 ready 后再启动；客户端入口 http://<ip>:7080/v1
```

- 请求里 `model` 填 `Qwen/Qwen3.8-Flash-Next`（`SERVED_MODEL_NAME` 可改）。
- 启动参数不带 `--sampling-defaults openai`，采样参数默认取模型的 generation_config；建议请求里 `top_k <= 32`，例如 `{"top_k": 20, "top_p": 0.95}`。
- MTP 默认 `SPECULATIVE_NUM_STEPS=3`；改 2 时设置环境变量即可。
- 关闭 P 端 radix：`bash ./start_prefill.sh --disable-radix-cache`。
- 单节点、无 RDMA 时 Mooncake 走 TCP（`pd_env.sh` 中 `MOONCAKE_PROTOCOL=tcp`）。跨节点 RDMA 部署需另行配置。

## 验证结果（MI308X，单节点 1P1D，MTP step=3，P 端 radix 开）

在 `qwen38-flash-next-20261102` 镜像上测得。功能与精度（ATOM `qwen38-pd-fix`，即 v2 之前的版本）：

| 检查项 | 结果 |
|---|---|
| 确定性文本题 6 道 | 6/6 |
| 单图多模态、同图重发 | 正确 |
| 真实 agent 请求（65.7K token、30 张图）冷启动 2 次 + 重发 | 均 `finish_reason=stop`，重发时 P 端 radix 命中 65,664 token |
| GSM8K 400 题（关 thinking） | 0.9675 ~ 0.98 |
| IFBench 300 题（thinking，T=1.0，top_k=20，top_p=0.95，32K） | prompt-strict 79.33；截断 3.3%，逐字循环 1 条 |

本分支（v2）：文本题 6/6；GSM8K 0.9775（EP2）/ 0.975（EP1）。

单请求 TPOT（输入 6.5K，其中 5K 共享前缀、P 端命中 4,992 token；输出 400；并发 1；每组 20 个请求 × 2 轮）：

| ATOM | EP | 默认采样 TPOT 平均 (ms) | greedy TPOT 平均 (ms) | TTFT 中位 (ms) |
|---|---|---|---|---|
| `qwen38-pd-fix` | 2 | 5.92 / 6.11 | 5.87 / 6.03 | 265 ~ 271 |
| `qwen38-pd-fix-v2` | 2 | 6.00 / 6.12 | 5.79 / 6.11 | 269 ~ 282 |
| `qwen38-pd-fix-v2` | 1 | 5.95 / 5.90 | 5.86 / 5.74 | 292 ~ 305 |

## 已知限制

- 只验证了单节点 1P1D、P/D 同为 TP2、PP=1。
- D 端不能开 radix（SGLang 中 D 端 radix 与投机解码互斥）。
- `start_router.sh` 不再传 `--prepare-workers`（镜像内 atomesh 0.2.0 不支持）。
