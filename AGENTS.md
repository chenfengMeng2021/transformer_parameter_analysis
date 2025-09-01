## AGENTS

简要说明：本项目通过下载与读取大模型参数，并用 Jupyter Notebook 进行结构化分析（embedding、各层矩阵维度与秩、统计量等）。本文档约定了代理（Agent）工作流、规划模板、代码风格与产物格式，确保流程可复现、输出一致。

### 核心原则
- **先计划，后执行**：每个任务在动手前写出最小可行计划与完成标准。
- **最小必要抽象**：代码保持简洁直白，避免过度工程化；只在消除重复或隔离变更时抽象。
- **可复现与可追踪**：固定随机种子、记录版本与环境、保存中间产物和日志。
- **安全与节省资源**：按需下载、缓存模型，避免一次性加载全部权重到内存。

### 角色与职责
- **Orchestrator（编排）**：串联下载→读取→分析→导出，管理配置与产物路径。
- **Downloader（下载）**：从 Hugging Face/本地源下载权重并校验、缓存。
- **Reader（读取）**：在不训练场景下安全加载参数，按模块迭代权重张量。
- **Analyzer（分析）**：计算形状、层号、所属子模块、秩、均值、方差等；导出为表格/JSON。

### 目录与路径（建议）
- `data/models/`：本地模型缓存根目录
- `data/outputs/`：分析结果（CSV/JSON/Parquet）
- `notebooks/`：Jupyter 分析与可视化
- `scripts/`：可复用脚本（下载/读取/分析）

### 环境与工具
- Python 3.10+；在执行任何 Python 命令前，先激活项目虚拟环境：
```bash
source .venv/bin/activate
```
- 优先使用 `transformers`/`safetensors`/`bitsandbytes`（可选）进行只读加载。
- 建议设置 HF 缓存：`HF_HOME` 或默认 `~/.cache/huggingface`。

### 任务模板（Plan-Before-Execute）
每个任务（下载/读取/分析）至少包含：
- **目标**：要产生的具体产物与验收标准
- **输入**：模型标识、路径、设备、精度、数据切分等
- **步骤**：最小可行步骤列表，包含回退/重试策略
- **输出**：文件路径、格式、关键字段
- **校验**：尺寸/统计量/条目数、采样检查
- **日志**：记录耗时、错误、版本信息

示例（Downloader）：
```text
目标: 将 meta-llama/Llama-2-7b-hf 下载到 data/models/ 下并校验完整性
输入: model_id=meta-llama/Llama-2-7b-hf, revision=main
步骤: 1) 确认缓存 2) 分片并发下载 3) 校验哈希 4) 记录版本
输出: data/models/meta-llama/Llama-2-7b-hf/
校验: 关键分片数量、总大小、哈希比对
日志: download_{timestamp}.log
```

### 数据与输出规范
- **参数标识**：`model_path`, `param_name`, `layer_index`, `block_type`（如 embedding/q_proj/k_proj/v_proj/o_proj/moe/output）、`device`、`dtype`
- **形状信息**：`shape`（字符串），`rows`, `cols`（如适用）
- **统计量**：`rank`（近似或精确）、`mean`, `std`, `var`, `sparsity`（可选）
- **元信息**：`model_id`, `revision`, `framework`（pt/gguf/npz 等）, `created_at`, `code_version`
- **文件格式**：默认输出 `CSV`；可选 `JSONL/Parquet`（大规模优先 `Parquet`）

CSV 示例列：
```text
model_path,model_id,revision,param_name,layer_index,block_type,shape,rows,cols,rank,mean,std,var,sparsity,dtype,created_at
```

### 稀疏度（Sparsity）说明
- **定义**：稀疏度指张量中“接近零”的元素比例，范围在 [0, 1]。
- **默认阈值**：`abs(x) <= 1e-8` 视为零，可按需调整以适配不同精度。
- **直觉**：稀疏度越高，表示更多元素接近零；在压缩与加速场景中可用于剪枝、稀疏计算优化等分析参考。

### 代码风格（简洁、必要抽象）
- **命名清晰**：函数用动词短语，变量用名词短语；避免缩写；显式参数。
- **控制流**：优先早返回，先处理边界与错误；不要深层嵌套。
- **错误处理**：只在需要的地方捕获异常；带上上下文（模型/层/参数）。
- **注释**：只解释"为什么"，不赘述"怎么做"；保持简短。
- **模块化**：将下载、读取、分析拆成独立小函数；避免隐式全局状态。

### 项目工程风格
- **文件组织**：按功能模块组织，避免过深的目录嵌套；相关功能放在同一目录下。
- **命名约定**：目录和文件使用小写字母和下划线，如`data_processing/`、`model_analysis.py`。
- **配置管理**：使用单一配置文件或环境变量，避免硬编码路径和参数。
- **依赖管理**：明确指定版本范围，使用`requirements.txt`或`pyproject.toml`；虚拟环境隔离。
- **日志记录**：统一的日志格式和级别，包含时间戳、模块名、操作类型。
- **错误处理**：优雅降级，提供有意义的错误信息和恢复建议。
- **测试策略**：关键功能包含单元测试，使用pytest框架；测试数据与生产数据分离。
- **文档维护**：代码变更时同步更新相关文档；README包含快速开始指南。
- **版本控制**：有意义的提交信息，使用语义化版本号；重要变更记录在CHANGELOG中。
- **代码审查**：关键功能变更需要代码审查；保持代码风格一致性。
- **脚本管理**：不要增加含有相同部分或功能重叠的scripts，除非额外提及；只在对应的script内部修改，避免重复实现。

### 标准实现片段
- 下载（Hugging Face）：
```python
from huggingface_hub import snapshot_download

def download_model(model_id: str, local_dir: str, revision: str = None) -> str:
    return snapshot_download(repo_id=model_id, local_dir=local_dir, revision=revision, local_dir_use_symlinks=False)
```

- 只读加载与参数遍历（PyTorch Transformers 示例）：
```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig

@torch.inference_mode()
def iterate_parameters(model_path: str):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
    for name, param in model.named_parameters():
        yield name, param.detach().to("cpu")
```

- 统计与近似秩（加入稀疏度）：
```python
import numpy as np

def tensor_stats(x: np.ndarray, rank_k: int = 128, zero_epsilon: float = 1e-8):
    x2d = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
    u, s, vh = np.linalg.svd(x2d[:,:min(x2d.shape)], full_matrices=False)
    r = int((s > 1e-6 * s.max()).sum()) if s.size else 0
    num_total = x.size
    num_near_zero = int((np.abs(x) <= zero_epsilon).sum()) if num_total else 0
    sparsity = float(num_near_zero) / num_total if num_total else 0.0
    return {
        "shape": str(tuple(x.shape)),
        "rows": int(x2d.shape[0]),
        "cols": int(x2d.shape[1]),
        "rank": int(min(r, rank_k)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "var": float(x.var()),
        "sparsity": float(sparsity),
        "dtype": str(x.dtype),
    }
```

- 模块归类（embedding / qkvo / moe / output）：
```python
def classify_param(name: str) -> tuple[str, int | None]:
    lname = name.lower()
    layer_idx = None
    if ".layers." in lname:
        try:
            layer_idx = int(lname.split(".layers.")[1].split(".")[0])
        except Exception:
            layer_idx = None
    if "embed" in lname:
        return "embedding", layer_idx
    if ".q_proj" in lname:
        return "q_proj", layer_idx
    if ".k_proj" in lname:
        return "k_proj", layer_idx
    if ".v_proj" in lname:
        return "v_proj", layer_idx
    if ".o_proj" in lname:
        return "o_proj", layer_idx
    if "moe" in lname or "gate" in lname or "expert" in lname:
        return "moe", layer_idx
    if "lm_head" in lname or "output" in lname:
        return "output", layer_idx
    return "other", layer_idx
```

### Orchestrator 参考流程
1) 计划：填写任务模板（模型 id、输出路径、校验标准）
2) 下载：如已存在且校验通过则跳过
3) 读取：以只读/半精度加载，逐参数迭代
4) 分析：计算统计量与秩，归类模块、提取层号
5) 导出：写入 `CSV/JSONL/Parquet`；保存 `run_meta.json`
6) 校验：行数、列名、缺失值、随机抽查若干条

### Jupyter 分析流程（建议 Notebook 结构）
- `00_setup.ipynb`：环境、路径、依赖检查
- `10_summary_stats.ipynb`：读取导出表，做汇总与可视化（直方图/箱线图）
- `20_per_layer.ipynb`：按层统计维度与秩，定位异常层
- `30_embedding_output.ipynb`：聚焦 embedding 与 lm_head

常用单元片段：
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("data/outputs/params.parquet")
sns.histplot(df[df.block_type=="o_proj"]["rank"], bins=50)
plt.show()
```

### 命令与执行
- 激活环境：
```bash
source .venv/bin/activate
```
- 下载：
```bash
python scripts/download.py --model_id meta-llama/Llama-2-7b-hf --out data/models/
```
- 分析导出：
```bash
python scripts/analyze.py --model_path data/models/meta-llama/Llama-2-7b-hf \
  --out data/outputs/params.csv --format csv
```

### 验收清单（Checklist）
- 已写明计划与完成标准
- 下载成功且通过哈希/完整性校验
- 读取遍历覆盖全部可学习参数（或明确过滤策略）
- 输出包含：模型路径、层号、模块类型、形状、秩、mean/variance、sparsity 等
- 产物格式正确，无缺失关键列；随机抽检数值合理
- 记录版本与环境（依赖清单、代码提交号、时间戳）

### 版本与溯源
- 保存 `run_meta.json`：`model_id`、`revision`、`code_version`、`python/torch/transformers` 版本、时间戳、命令行参数。
- 固定随机种子；记录设备与精度；确保结果可复现。

### 故障与回退
- 下载失败：切换镜像/重试分片/改 revision；尽可能复用缓存。
- 内存不足：只读分片、`device_map="auto"`、降低精度、分模块分析。
- 维度异常：记录参数名与形状，跳过该条并写错误日志。

---
如需扩展（如添加更多统计量或支持其他框架），优先按“最小必要抽象”加小函数与小脚本，保持接口与数据列稳定。
