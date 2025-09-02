# Transformer Parameter Analysis

A research toolkit for downloading, reading, and analyzing large language model parameters. It downloads and reads model weights and performs scripted, structured analysis (embeddings, per-layer matrix dimensions and ranks, statistics, etc.).

## 🚀 Features

- Model download: Automatically download model weights from Hugging Face
- Parameter analysis: Analyze parameter shapes, ranks, and statistics
- Embedding analysis: Specialized embedding-layer analysis, including K-means clustering
- Transformer analysis: Analyze per-layer transformer matrices, including SVD-based singular value analysis
- GPU acceleration: NVIDIA GPU support for large-matrix SVD
- Data export: CSV, JSON, and more
- Visualization: Plots and summary reports

## 📋 System Requirements

- Python 3.10+
- Supported OS: Linux, macOS, Windows
- Recommended RAM: 8GB+ (depends on model size)
- GPU support: NVIDIA GPU with CUDA (RTX 3080+ recommended)
- VRAM: at least 8GB (16GB+ recommended)

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone git@github.com:chenfengMeng2021/transformer_parameter_analysis.git
cd transformer_parameter_analysis
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt

# For GPU support, install the CUDA build of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📁 Project Structure

```
transformer_parameter_analysis/
├── data/                          # Data directory
│   ├── models/                    # Model cache
│   └── outputs/                   # Analysis outputs
├── scripts/                       # Core scripts
│   ├── download.py               # Model download script
│   ├── analyze.py                # Parameter analysis script
│   ├── embedding_analysis.py     # Embedding analysis script
│   └── transformer_analysis.py   # Transformer analysis script (GPU supported)
├── run.py                        # One-click entry (repo root): download + analyze + report
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # Project description
```

## 🚀 Quick Start

### 0. One-click pipeline (download → analyze → report)
```bash
# Activate virtual environment
source .venv/bin/activate

# One command: auto-download the model, run embedding and transformer analyses, and generate a report
python run.py --model_id "Qwen/Qwen3-4B" --output_dir data/outputs

# CPU mode (disable GPU)
python run.py --model_id "Qwen/Qwen3-4B" --output_dir data/outputs --use_cpu
```

Main outputs:
- Transformer CSV: `data/outputs/transformer_matrix_analysis.csv`
- Embedding ANOVA curve: `data/outputs/anova_curve.png`
- Embedding clustering summary: `data/outputs/clustering_results.json`
- Report: `data/outputs/report.md`

### 1. Download a model
```bash
# Activate virtual environment
source .venv/bin/activate

# Download to local cache
python scripts/download.py --model_id Qwen/Qwen3-4B --out data/models/
```

### 2. Analyze model parameters
```bash
# Analyze parameters and export as CSV
python scripts/analyze.py --model_path data/models/Qwen/Qwen3-4B --out data/outputs/params.csv --format csv
```

### 3. Transformer matrix analysis (recommended)
```bash
# Analyze transformer matrices with GPU acceleration
python scripts/transformer_analysis.py --model_path data/models --out data/outputs/transformer_analysis.csv
```

### 4. Embedding analysis
```bash
# Analyze embedding layers and perform clustering (outputs to data/outputs)
python scripts/embedding_analysis.py --model_path data/models/Qwen/Qwen3-4B --output_dir data/outputs
```

## 🚀 GPU Acceleration

### New capabilities
- Full-matrix analysis: analyze full parameter matrices without sampling
- CUDA acceleration: use PyTorch's CUDA backend for SVD
- Smart memory management: automatically handle GPU OOM
- Graceful degradation: fall back to CPU when GPU memory is insufficient

### GPU vs CPU speedup
| Matrix size | CPU time | GPU time | Speedup |
|-------------|----------|----------|---------|
| 512×512     | ~0.5s    | ~0.1s    | 5x      |
| 1024×1024   | ~2s      | ~0.3s    | 7x      |
| 2048×2048   | ~8s      | ~1s      | 8x      |
| 4096×4096   | ~32s     | ~4s      | 8x      |

### Using the GPU
```bash
# Use GPU acceleration (default)
python scripts/transformer_analysis.py --model_path data/models --out results.csv

# Force CPU analysis
python scripts/transformer_analysis.py --model_path data/models --out results.csv --no_gpu
```

## 📊 Output Formats

### Transformer analysis output (CSV)
Columns include:
- `model_path`: model path
- `model_id`: model identifier
- `revision`: model revision
- `param_name`: parameter name
- `layer_index`: layer index
- `block_type`: module type (embedding/q_proj/k_proj/v_proj/o_proj/moe/output)
- `shape`: parameter shape
- `rows`, `cols`: rows and columns
- `design_rank`: design rank
- `actual_rank`: numerical rank
- `rank_ratio`: rank ratio
- `svd_90_percent_rank`: number of singular values to explain 90% variance
- `svd_90_percent_ratio`: ratio of 90% rank to design rank
- `mean`, `std`, `var`: statistics
- `sparsity`: sparsity
- `dtype`: data type
- `created_at`: creation time

Note: CSV output is sorted by numeric `layer_index` ascending; non-layer parameters (e.g., embedding, output head) are placed at the end.

### Console output example
```
================================================================================
LAYER-BY-LAYER MATRIX ANALYSIS
================================================================================

🔹 LAYER 0
--------------------------------------------------
  📊 q_proj        | Shape: (512, 512)      | Rank: 128/512 (25.0%)
      SVD 90%:  64/512 (12.5%) | Sparsity: 10.0% | μ= 0.0000 σ= 0.1000
  📊 k_proj        | Shape: (512, 512)      | Rank: 128/512 (25.0%)
      SVD 90%:  64/512 (12.5%) | Sparsity: 10.0% | μ= 0.0000 σ= 0.1000
```

## 🔧 Scripts

### transformer_analysis.py (recommended)
New features: SVD singular value analysis and GPU acceleration
- Analyze all transformer matrix parameters
- Compute the number of singular values for 90% variance
- GPU-accelerated SVD
- Pretty-printed per-layer matrix summary

Args:
- `--model_path`: model path (required)
- `--out`: output CSV path (required)
- `--model_id`: model id (optional)
- `--revision`: model revision (optional)
- `--no_gpu`: disable GPU acceleration

### run.py (one-click entry)
Download model → run Transformer/Embedding analyses → generate a report.

Args:
- `--model_id`: Hugging Face model ID (required)
- `--revision`: model revision/commit (optional)
- `--output_dir`: output directory (default `data/outputs`)
- `--use_cpu`: force CPU mode (optional)

### download.py
Model download script supporting:
- Downloading from Hugging Face
- Specifying versions and branches
- Automatic cache management

Args:
- `--model_id`: Hugging Face model ID (required)
- `--out`: output directory (required)
- `--revision`: model revision (optional)

### analyze.py
General parameter analysis script supporting:
- Automatic detection of embedding weights
- Statistics and rank computation
- Multiple output formats

Args:
- `--model_path`: model path (required)
- `--out`: output file path (required)
- `--format`: output format (csv/json) (optional)

### embedding_analysis.py
Dedicated embedding-layer analysis script supporting:
- K-means clustering
- ANOVA-based optimal k selection (one-way ANOVA to evaluate significance across k)
- Visualization

Args:
- `--model_path`: model path (required)
- `--out`: output file path (required)
- `--random_state`: random seed (optional)

## 📈 Analysis Features

### SVD singular value analysis (new)
- 90% variance: count how many singular values can represent over 90% of the matrix variance
- Compressibility: identify matrices suitable for low-rank approximation
- Performance optimization: locate bottlenecks and opportunities

### Statistics
- Shape analysis: automatically detect parameter dimensions
- Rank analysis: design rank vs numerical rank
- Statistical features: mean, standard deviation, variance
- Sparsity: proportion of near-zero elements

### Module classification
Parameters are automatically classified as:
- `embedding`: token embeddings
- `q_proj`: Query projection
- `k_proj`: Key projection
- `v_proj`: Value projection
- `o_proj`: Output projection
- `moe`: Mixture-of-Experts
- `norm`: Normalization
- `output`: Output layer

### Clustering
- Use ANOVA (one-way) to choose the optimal number of clusters
- K-means clustering
- Result visualization

## 🔍 Examples

### End-to-end: Qwen3-4B example (download → analyze → report)
```bash
# 1) Activate environment
source .venv/bin/activate

# 2) Download the model (cached under data/models/Qwen-Qwen3-4B/)
python scripts/download.py --model_id Qwen/Qwen3-4B --out data/models/

# 3) Run transformer matrix analysis (CSV under data/outputs/Qwen-Qwen3-4B/)
python scripts/transformer_analysis.py \
  --model_path data/models \
  --out data/outputs/Qwen-Qwen3-4B/transformer_matrix_analysis.csv \
  --model_id "Qwen/Qwen3-4B"

# 4) Run embedding analysis (clusters, ANOVA curve, embedding CSV)
python scripts/embedding_analysis.py \
  --model_path data/models/Qwen-Qwen3-4B \
  --output_dir data/outputs/Qwen-Qwen3-4B

# 5) Generate a summary report
python run.py --model_id "Qwen/Qwen3-4B" --output_dir data/outputs/Qwen-Qwen3-4B
```

Outputs for this example (already present in the repo under `data/outputs/Qwen-Qwen3-4B/`):
- `transformer_matrix_analysis.csv`
- `qwen3-4b_embedding.csv`
- `clustering_results.json`
- `cluster_representatives.json`
- `anova_curve.png`
- `report.md`

### Analyze Llama-2
```bash
# Download the model
python scripts/download.py --model_id meta-llama/Llama-2-7b-hf --out data/models/

# Analyze with GPU acceleration
python scripts/transformer_analysis.py --model_path data/models --out data/outputs/llama2_analysis.csv
```



## 🐛 Troubleshooting

### Common issues

1. GPU out-of-memory
   ```
   ⚠️  GPU OOM, falling back to CPU for matrix (4096, 4096)
   ```
   Solutions:
   - Use `--no_gpu` to force CPU mode
   - Close other GPU applications

2. CUDA not available
   ```
   ❌ CUDA not available
   ```
   Solutions:
   - Check NVIDIA driver installation
   - Verify PyTorch CUDA version
   - Confirm GPU hardware support

3. Insufficient memory
   - Use `--low_cpu_mem_usage`
   - Lower precision (float16)
   - Process large models in batches

4. Download failure
   - Check network connection
   - Use mirror sources
   - Retry download

5. Unsupported model format
   - Ensure the model supports the safetensors format
   - Check model structure

### Logs and debugging
- All scripts print detailed logs
- Use `--verbose` for more debug info

## 🤝 Contributing

Contributions are welcome! Please follow these principles:

1. Plan before execution: write a minimal viable plan first
2. Minimal necessary abstraction: keep the code simple and avoid over-engineering
3. Reproducibility: fix random seeds and record versions and environment information
4. Documentation: add docs and examples for new features

## 📄 License

This project uses the MIT License.

## 🙏 Acknowledgments

- Hugging Face team for `transformers`
- PyTorch team for the deep learning framework
- Open-source community support and contributions

## 📞 Contact

For issues or suggestions:
- Open an Issue
- Email
- Join discussions

---

Note: This project is for research and learning purposes only. Please comply with the usage terms and licenses of the respective models.
