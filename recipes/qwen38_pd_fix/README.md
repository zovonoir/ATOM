# Qwen3.8-Flash-Next 1P1D PD 分离（基于 qwen38-flash-next-20261102 镜像）

在已有的 `atom-sglang:qwen38-flash-next-20261102` 镜像中，把 ATOM / SGLang / aiter 三个组件换成
`qwen38-pd-fix` 分支，即可跑 SGLang + ATOM 的单节点 1P1D（P、D 各 TP2/EP2，MTP，P 端 radix，纯文本与多模态）。
不需要新镜像。

## 分支内容

| 组件 | 分支 | 基线 | 分支上的改动 |
|---|---|---|---|
| ATOM | `zovonoir/ATOM@qwen38-pd-fix` | ROCm/ATOM `f3d0b30` | `qwen4_exp_pd.py`：把插件自有的 QSA indexer（raw/compressed）和 PLE 状态登记进 SGLang PD 传输；`ATOM_FLASH_PD_ALLOW_RADIX=1` 允许 P 端开 radix；本目录脚本 |
| SGLang | `zovonoir/sglang@qwen38-pd-fix` | sgl-project/sglang `baa9bef`（v0.5.22） | ROCm pyproject；MTP 开启后思考过程死循环的修复（HIP 上按采样参数做 EAGLE 验证，不做全词表排序，即 v0.5.20 交付时 `1a9977fe` 补丁的移植）；PD 首步稀疏 draft proposal |
| aiter | `zovonoir/aiter@qwen38-pd-fix` | ROCm/aiter `fb3f506`（composable_kernel `f93c029`） | 无改动 |

另外需要 `flydsl==0.5.3`（镜像里是 0.4.1，新 aiter 要求 >= 0.5）。镜像里的 PyTorch / Triton / Mooncake / atomesh 不变。

## 安装（容器内执行一次）

```bash
git clone --branch qwen38-pd-fix --single-branch https://github.com/zovonoir/ATOM.git /opt/qwen38-pd-fix/ATOM
bash /opt/qwen38-pd-fix/ATOM/recipes/qwen38_pd_fix/install.sh
```

`install.sh` 会：克隆 SGLang、aiter（含 composable_kernel 子模块）分支；`pip install flydsl==0.5.3`；
把 `/app/ATOM`、`/app/sglang`、`/app/aiter-test` 替换为分支代码（原目录保留为 `/app/*.orig-<时间戳>`）；
清掉 aiter 的 JIT 产物，首次启动时为 gfx942 重新编译（约 15 分钟，之后复用）。

无外网时：在别处把三个分支 clone 到同一目录（aiter 带 `--recurse-submodules`）并下载 flydsl 0.5.3 的 wheel，拷进容器后执行
`SRC=<该目录> FLYDSL_WHEEL=<wheel 路径> bash install.sh`。

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

在 `qwen38-flash-next-20261102` 镜像 + 本分支上测得：

| 检查项 | 结果 |
|---|---|
| 确定性文本题 6 道 | 6/6 |
| 单图多模态、同图重发 | 正确 |
| 真实 agent 请求（65.7K token、30 张图）冷启动 2 次 + 重发 | 均 `finish_reason=stop`，重发时 P 端 radix 命中 65,664 token |
| GSM8K 400 题（关 thinking） | 0.9675 ~ 0.98 |
| IFBench 300 题（thinking，T=1.0，top_k=20，top_p=0.95，32K） | prompt-strict 79.33；截断 3.3%，逐字循环 1 条 |

## 已知限制

- 只验证了单节点 1P1D、P/D 同为 TP2、PP=1。
- D 端不能开 radix（SGLang 中 D 端 radix 与投机解码互斥）。
- `start_router.sh` 不再传 `--prepare-workers`（镜像内 atomesh 0.2.0 不支持）。
