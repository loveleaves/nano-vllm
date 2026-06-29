# 环境搭建指南（uv）

## 环境信息

| 项目 | 版本 |
|------|------|
| OS | Linux (WSL2, Ubuntu) |
| GPU | NVIDIA RTX 3060 Ti 8GB |
| CUDA Toolkit | 12.4 |
| Driver | 596.36 |
| Python | 3.12.13 |
| uv | 0.11.17 |

## 前置条件

- 已安装 [uv](https://github.com/astral-sh/uv)：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- CUDA Toolkit 已安装（需与 torch CUDA 版本匹配）
- 已下载模型权重（见下方）

## 模型准备

本机使用 Qwen3-1.7B，权重路径 `/home/cb/model/Qwen3-1.7B/`。

如需下载其他版本（以 Qwen3-0.6B 为例）：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/model/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm
```

### 2. 创建虚拟环境

```bash
uv venv .venv --python 3.12
```

uv 会自动下载 CPython 3.12（若系统未安装），创建 `.venv/` 目录。

### 3. 安装 PyTorch（CUDA 版本）

根据 CUDA Toolkit 版本选择对应的 wheel index：

| CUDA Toolkit | Index URL |
|-------------|-----------|
| 12.1 | `https://download.pytorch.org/whl/cu121` |
| 12.4 | `https://download.pytorch.org/whl/cu124` |
| 12.8 | `https://download.pytorch.org/whl/cu128` |

本机 CUDA 12.4，使用 cu121（ABI 兼容）：

```bash
uv pip install --python .venv torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

> **说明**：torch 会同时安装 triton（自动作为依赖）。

### 4. 安装常规依赖

```bash
uv pip install --python .venv transformers==4.57.6 xxhash tqdm
```

### 5. 编译安装 flash-attn

flash-attn 在 PyPI 只提供源码包，需要本地编译。先安装编译工具，再用 `--no-build-isolation` 复用已装的 torch 环境：

```bash
# 安装编译依赖
uv pip install --python .venv wheel setuptools

# 编译 flash-attn（MAX_JOBS 控制并行度，按 CPU 核数设置）
MAX_JOBS=$(nproc) uv pip install --python .venv flash-attn --no-build-isolation
```

> **耗时参考**：12 核 CPU 约 30 秒，4 核 CPU 约 2-5 分钟。  
> **原理**：`--no-build-isolation` 让 flash-attn 的 setup.py 直接使用 `.venv` 中已安装的 torch/CUDA，无需重新创建隔离构建环境。

### 6. 安装 nano-vllm 包

```bash
uv pip install --python .venv -e . --no-deps
```

`-e`（editable）模式：修改源码后无需重新安装即可生效。  
`--no-deps`：依赖已手动安装，避免重复解析。

### 7. 验证安装

```bash
source .venv/bin/activate
python3 -c "
import torch, flash_attn, transformers, nanovllm
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('flash_attn:', flash_attn.__version__)
print('transformers:', transformers.__version__)
print('nanovllm OK')
"
```

预期输出：

```
torch: 2.5.1+cu121 | CUDA: True
flash_attn: 2.8.3
transformers: 4.57.6
nanovllm OK
```

## 运行示例

修改 `example.py` 中的 `path` 为本机模型目录，然后运行：

```bash
source .venv/bin/activate
python3 example.py
```

成功输出示例（Qwen3-1.7B，enforce_eager=True）：

```
Generating: 100%|██████████| 2/2 [00:18, Prefill=6tok/s, Decode=18tok/s]

Prompt: '<|im_start|>user\nintroduce yourself...'
Completion: "...I'm Qwen, an AI assistant developed by Alibaba Cloud..."

Prompt: '<|im_start|>user\nlist all prime numbers within 100...'
Completion: "...2, 3, 5, 7, 11, 13..."
```

## 已安装包版本

| 包 | 版本 |
|----|------|
| torch | 2.5.1+cu121 |
| triton | 3.1.0 |
| flash-attn | 2.8.3 |
| transformers | 4.57.6 |
| xxhash | 3.7.0 |
| tqdm | 4.68.0 |
| wheel | 0.47.0 |
| nano-vllm | 0.2.0 (editable) |

## 常见问题

### flash-attn 编译失败：`No module named 'wheel'`

```bash
uv pip install --python .venv wheel setuptools
# 然后重新执行 flash-attn 安装命令
```

### flash-attn 编译失败：CUDA 版本不匹配

确保 `nvcc --version` 和 torch CUDA 版本一致。可以用 `nvidia-smi` 查看驱动支持的最高 CUDA 版本。

### `CUDA not available`（`torch.cuda.is_available()` 返回 False）

确认安装的是 CUDA 版 torch，而非 CPU 版：
```bash
python3 -c "import torch; print(torch.version.cuda)"
# 应输出 "12.1" 而非 "None"
```

### 内存不足（OOM）

减少 `max_num_batched_tokens` 或 `gpu_memory_utilization`：
```python
llm = LLM(path, gpu_memory_utilization=0.8, max_num_batched_tokens=8192)
```

## 完整命令汇总

```bash
# 一键安装（12核CPU）
uv venv .venv --python 3.12
uv pip install --python .venv torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv transformers==4.57.6 xxhash tqdm wheel setuptools
MAX_JOBS=$(nproc) uv pip install --python .venv flash-attn --no-build-isolation
uv pip install --python .venv -e . --no-deps

# 激活并运行
source .venv/bin/activate
python3 example.py
```
