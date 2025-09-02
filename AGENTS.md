## AGENTS

Brief: This project downloads and reads large-model parameters and performs structured analysis (embeddings, per-layer matrix dimensions and ranks, summary statistics, etc.). This document defines the Agent workflow, planning template, code style, and artifact formats to ensure reproducibility and consistent outputs.

### Core Principles
- Plan before executing: write a minimal viable plan and completion criteria for every task before coding.
- Minimal necessary abstraction: keep code simple and direct; abstract only to remove duplication or isolate change.
- Reproducible and traceable: fix random seeds, record versions and environment, persist intermediates and logs.
- Safe and resource-conscious: download on demand and cache models; avoid loading all weights into memory at once.

### Roles and Responsibilities
- Orchestrator: chains Download → Read → Analyze → Export; manages configs and artifact paths.
- Downloader: downloads weights from Hugging Face/local mirrors, verifies integrity, and caches.
- Reader: safely loads parameters in non-training mode and iterates tensors by module.
- Analyzer: computes shape, layer index, submodule type, rank, mean, variance, etc.; exports as tables/JSON.

### Recommended Directories and Paths
- `data/models/`: local model cache root
- `data/outputs/`: analysis results (CSV/JSON/Parquet)
- `notebooks/`: Jupyter analyses and visualizations
- `scripts/`: reusable scripts (download/read/analyze)

### Environment and Tools
- Python 3.10+. Before running any Python command, activate the project virtual environment:
```bash
source .venv/bin/activate
```
- Prefer `transformers`/`safetensors`/`bitsandbytes` (optional) for read-only loading.
- Configure HF cache via `HF_HOME` or the default `~/.cache/huggingface`.

### Task Template (Plan-Before-Execute)
Each task (download/read/analyze) must include at least:
- Goal: concrete deliverables and acceptance criteria
- Inputs: model identifier, path, device, precision, data split, etc.
- Steps: minimal viable steps including rollback/retry strategy
- Outputs: file paths, formats, key fields
- Validation: sizes/statistics/row counts, sampled checks
- Logging: record duration, errors, versions

Example (Downloader):
```text
Goal: Download meta-llama/Llama-2-7b-hf to data/models/ and verify integrity
Inputs: model_id=meta-llama/Llama-2-7b-hf, revision=main
Steps: 1) Check cache 2) Parallel shard download 3) Verify hashes 4) Record version
Outputs: data/models/meta-llama/Llama-2-7b-hf/
Validation: number of shards, total size, hash match
Logs: download_{timestamp}.log
```

### Data and Output Specification
- Parameter identifiers: `model_path`, `param_name`, `layer_index`, `block_type` (e.g., embedding/q_proj/k_proj/v_proj/o_proj/moe/output), `device`, `dtype`
- Shape info: `shape` (string), `rows`, `cols` (when applicable)
- Statistics: `rank` (approx or exact), `mean`, `std`, `var`, `sparsity` (optional)
- Metadata: `model_id`, `revision`, `framework` (pt/gguf/npz, etc.), `created_at`, `code_version`
- File formats: default `CSV`; optional `JSONL/Parquet` (prefer `Parquet` for large scale)

CSV example columns:
```text
model_path,model_id,revision,param_name,layer_index,block_type,shape,rows,cols,rank,mean,std,var,sparsity,dtype,created_at
```

Note: Exported CSVs are sorted by numeric `layer_index` ascending; non-layer parameters (e.g., `embedding`, `lm_head`) go to the end of the file.

### Sparsity Definition
- Definition: proportion of elements “near zero” in a tensor, in [0, 1].
- Default threshold: `abs(x) <= 1e-8` is considered zero; adjust as needed for different precisions.
- Intuition: higher sparsity means more elements are near zero; useful for pruning and sparse compute optimization.

### Code Style (Simple, Necessary Abstractions)
- Clear naming: functions use verb phrases; variables use noun phrases; avoid abbreviations; explicit parameters.
- Control flow: prefer early returns; handle edges and errors first; avoid deep nesting.
- Error handling: catch exceptions only where needed; include context (model/layer/parameter).
- Comments: explain why, not how; keep them brief.
- Modularity: split download/read/analyze into small functions; avoid implicit global state.

### Project Engineering Style
- File organization: group by functionality; avoid deep directory nesting; colocate related features.
- Naming convention: lowercase with underscores for dirs/files, e.g., `data_processing/`, `model_analysis.py`.
- Configuration management: use a single config file or environment variables; avoid hardcoded paths/params.
+- Dependency management: specify version ranges explicitly using `requirements.txt` or `pyproject.toml`; isolate with a virtual environment.
- Logging: unified log format and levels with timestamp, module, operation type.
- Error handling: graceful degradation with actionable errors and recovery suggestions.
- Testing strategy: unit tests for critical functions using pytest; separate test data from production data.
- Documentation: update docs alongside code changes; README includes a quickstart.
- Version control: meaningful commit messages; semantic versioning; important changes documented in CHANGELOG.
- Script management: do not add overlapping/duplicative scripts unless explicitly required; modify within the corresponding script to avoid duplication.

### Standard Implementation Snippets
- Download (Hugging Face):
```python
from huggingface_hub import snapshot_download

def download_model(model_id: str, local_dir: str, revision: str | None = None) -> str:
    return snapshot_download(repo_id=model_id, local_dir=local_dir, revision=revision, local_dir_use_symlinks=False)
```

- Read-only loading and parameter iteration (PyTorch Transformers example):
```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig

@torch.inference_mode()
def iterate_parameters(model_path: str):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    for name, param in model.named_parameters():
        yield name, param.detach().to("cpu")
```

- Statistics and rank (with sparsity):
```python
import numpy as np

def tensor_stats(x: np.ndarray, zero_epsilon: float = 1e-8):
    x2d = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
    # Full SVD for numerical rank; no artificial cap on rank
    _, s, _ = np.linalg.svd(x2d[:, : min(x2d.shape)], full_matrices=False)
    r = int((s > 1e-6 * s.max()).sum()) if s.size else 0
    num_total = x.size
    num_near_zero = int((np.abs(x) <= zero_epsilon).sum()) if num_total else 0
    sparsity = float(num_near_zero) / num_total if num_total else 0.0
    return {
        "shape": str(tuple(x.shape)),
        "rows": int(x2d.shape[0]),
        "cols": int(x2d.shape[1]),
        "rank": int(r),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "var": float(x.var()),
        "sparsity": float(sparsity),
        "dtype": str(x.dtype),
    }
```

- Module classification (embedding / qkvo / moe / output):
```python
def classify_param(name: str) -> tuple[str, int | None]:
    lname = name.lower()
    layer_index = None
    if ".layers." in lname:
        try:
            layer_index = int(lname.split(".layers.")[1].split(".")[0])
        except Exception:
            layer_index = None
    if "embed" in lname:
        return "embedding", layer_index
    if ".q_proj" in lname:
        return "q_proj", layer_index
    if ".k_proj" in lname:
        return "k_proj", layer_index
    if ".v_proj" in lname:
        return "v_proj", layer_index
    if ".o_proj" in lname:
        return "o_proj", layer_index
    if "moe" in lname or "gate" in lname or "expert" in lname:
        return "moe", layer_index
    if "lm_head" in lname or "output" in lname:
        return "output", layer_index
    return "other", layer_index
```

### Orchestrator Reference Flow
1) Plan: fill in the template (model id, output path, validation criteria)
2) Download: skip if already present and validated
3) Read: load in read-only/half precision, iterate parameters
4) Analyze: compute statistics and rank, classify modules, extract layer indices
5) Export: write `CSV/JSONL/Parquet`; save `run_meta.json`
6) Validate: row counts, column names, missing values, random spot checks

### Jupyter Analysis Flow (Suggested Notebook Structure)
- `00_setup.ipynb`: environment, paths, dependency checks
- `10_summary_stats.ipynb`: read exported tables; summary and visualization (hist/box plots)
- `20_per_layer.ipynb`: per-layer dimensions and ranks; locate anomalous layers
- `30_embedding_output.ipynb`: focus on embedding and lm_head

Common notebook snippet:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet("data/outputs/params.parquet")
sns.histplot(df[df.block_type=="o_proj"]["rank"], bins=50)
plt.show()
```

### Commands and Execution
- Activate environment:
```bash
source .venv/bin/activate
```
- Download:
```bash
python scripts/download.py --model_id meta-llama/Llama-2-7b-hf --out data/models/
```
- Analyze and export:
```bash
python scripts/analyze.py --model_path data/models/meta-llama/Llama-2-7b-hf \
  --out data/outputs/params.csv --format csv
```

- Transformer parameter analysis script (sorted by numeric layer index; `rank_k` removed):
```bash
python scripts/transformer_analysis.py \
  --model_path data/models \
  --out data/outputs/qwen3_matrix_analysis.csv \
  --model_id "Qwen/Qwen3-4B"
```

### Acceptance Checklist
- Plan and completion criteria documented
- Download succeeded and passed hash/integrity checks
- Parameter iteration covers all learnable parameters (or filtering is explicit)
- Output includes: model path, layer index, module type, shape, rank, mean/variance, sparsity, etc.
- File format is correct with no missing key columns; sampled values look reasonable
- Versions and environment recorded (dependency list, commit hash, timestamp)

### Versioning and Provenance
- Save `run_meta.json`: `model_id`, `revision`, `code_version`, `python/torch/transformers` versions, timestamp, CLI args.
- Fix random seeds; record device and precision; ensure reproducibility.

### Failures and Rollback
- Download failure: switch mirrors/retry shards/change revision; reuse cache when possible.
- OOM: read-only sharding, `device_map="auto"`, lower precision, analyze by module.
- Dimension issues: record parameter name and shape; skip the item and log an error.

---
For extensions (e.g., more statistics or additional frameworks), apply the “minimal necessary abstraction” rule: add small functions and scripts while keeping interfaces and data columns stable.
