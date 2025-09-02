# AI Block Research

A research toolkit for downloading, reading, and analyzing large language model parameters. This project downloads and reads large model parameters, and performs structured analysis (embedding, matrix dimensions and rank of each layer, statistics, etc.).

## 🚀 Features

- **Model Download**: Automatically download model weight files from Hugging Face
- **Parameter Analysis**: Analyze model parameter shapes, ranks, statistics, etc.
- **Embedding Analysis**: Specialized embedding layer analysis, including K-means clustering
- **Transformer Analysis**: Analyze parameter characteristics of transformer layers, including SVD singular value analysis
- **GPU Acceleration**: Support NVIDIA GPU acceleration for large matrix SVD calculations
- **Data Export**: Support CSV, JSON and other output formats
- **Visualization**: Generate analysis charts and statistical reports

## 📋 System Requirements

- Python 3.10+
- Supported operating systems: Linux, macOS, Windows
- Recommended memory: 8GB+ (depending on model size)
- **GPU Support**: NVIDIA GPU with CUDA support (recommended RTX 3080+)
- **VRAM**: At least 8GB (recommended 16GB+)

## 🛠️ Installation

### 1. Clone the project
```bash
git clone git@github.com:chenfengMeng2021/transformer_parameter_analysis.git
cd transformer_parameter_analysis
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt

# If GPU support is needed, install CUDA version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📁 Project Structure

```
transformer_parameter_analysis/
├── data/                          # Data directory
│   ├── models/                    # Model cache directory
│   └── outputs/                   # Analysis result outputs
├── scripts/                       # Core scripts
│   ├── download.py               # Model download script
│   ├── analyze.py                # Parameter analysis script
│   ├── embedding_analysis.py     # Embedding layer analysis script
│   └── transformer_analysis.py   # Transformer analysis script (with GPU acceleration)
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # Project documentation
```

## 🚀 Quick Start

### 1. Download model
```bash
# Activate virtual environment
source .venv/bin/activate

# Download model to local
python scripts/download.py --model_id Qwen/Qwen3-4B --out data/models/
```

### 2. Analyze model parameters
```bash
# Analyze model parameters and export as CSV
python scripts/analyze.py --model_path data/models/Qwen/Qwen3-4B --out data/outputs/params.csv --format csv
```

### 3. Transformer matrix analysis (recommended)
```bash
# Use GPU acceleration to analyze transformer matrices
python scripts/transformer_analysis.py --model_path data/models --out data/outputs/transformer_analysis.csv
```

### 4. Embedding layer analysis
```bash
# Analyze embedding layer and perform clustering
python scripts/embedding_analysis.py --model_path data/models/Qwen/Qwen3-4B --out data/outputs/embedding_analysis.csv
```

## 🚀 GPU Acceleration Features

### New Features
- **Complete Matrix Analysis**: No longer requires sampling, analyzes complete matrix parameters
- **CUDA Acceleration**: Uses PyTorch's CUDA backend to accelerate SVD calculations
- **Smart Memory Management**: Automatically handles GPU memory insufficiency
- **Graceful Degradation**: Automatically falls back to CPU calculation when GPU memory is insufficient

### GPU vs CPU Performance Improvement
| Matrix Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 512×512     | ~0.5s    | ~0.1s    | 5x      |
| 1024×1024   | ~2s      | ~0.3s    | 7x      |
| 2048×2048   | ~8s      | ~1s      | 8x      |
| 4096×4096   | ~32s     | ~4s      | 8x      |

### GPU Usage
```bash
# Use GPU acceleration analysis (default)
python scripts/transformer_analysis.py --model_path data/models --out results.csv

# Force CPU analysis
python scripts/transformer_analysis.py --model_path data/models --out results.csv --no_gpu
```

## 📊 Output Format

### Transformer Analysis Output (CSV)
Contains the following fields:
- `model_path`: Model path
- `model_id`: Model identifier
- `revision`: Model version
- `param_name`: Parameter name
- `layer_index`: Layer index
- `block_type`: Module type (embedding/q_proj/k_proj/v_proj/o_proj/moe/output)
- `shape`: Parameter shape
- `rows`, `cols`: Number of rows and columns
- `design_rank`: Design rank
- `actual_rank`: Actual numerical rank
- `rank_ratio`: Rank ratio
- **`svd_90_percent_rank`**: Number of singular values corresponding to 90% variance (new feature)
- **`svd_90_percent_ratio`**: Ratio of 90% variance rank to design rank (new feature)
- `mean`, `std`, `var`: Statistics
- `sparsity`: Sparsity
- `dtype`: Data type
- `created_at`: Creation time

### Console Output Example
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

## 🔧 Script Descriptions

### transformer_analysis.py (recommended)
**New features**: SVD singular value analysis and GPU acceleration
- Analyze all transformer matrix parameters
- Calculate the number of singular values corresponding to 90% variance
- Support GPU acceleration for SVD calculations
- Format output of matrix information for each layer

**Parameters:**
- `--model_path`: Model path (required)
- `--out`: Output CSV file path (required)
- `--model_id`: Model ID (optional)
- `--revision`: Model version (optional)
- `--rank_k`: Maximum calculation rank (default: 128)
- `--no_gpu`: Disable GPU acceleration

### download.py
Model download script, supports:
- Download models from Hugging Face
- Specify version and branch
- Automatic cache management

**Parameters:**
- `--model_id`: Hugging Face model ID (required)
- `--out`: Output directory (required)
- `--revision`: Model version (optional)

### analyze.py
General parameter analysis script, supports:
- Automatic embedding weight recognition
- Calculate statistics and rank
- Multiple output formats

**Parameters:**
- `--model_path`: Model path (required)
- `--out`: Output file path (required)
- `--format`: Output format (csv/json) (optional)

### embedding_analysis.py
Specialized embedding layer analysis script, supports:
- K-means clustering analysis
- Elbow method to determine optimal number of clusters
- Result visualization

**Parameters:**
- `--model_path`: Model path (required)
- `--out`: Output file path (required)
- `--random_state`: Random seed (optional)

## 📈 Analysis Features

### SVD Singular Value Analysis (new feature)
- **90% Variance Analysis**: Calculate how many singular values can represent more than 90% of matrix variance
- **Compressibility Assessment**: Identify matrices that can be approximated with low rank
- **Performance Optimization**: Find computational bottlenecks and optimization opportunities

### Statistical Calculations
- **Shape Analysis**: Automatically identify parameter dimensions
- **Rank Analysis**: Calculate design rank vs actual rank of matrices
- **Statistical Features**: Mean, standard deviation, variance
- **Sparsity**: Calculate proportion of elements close to zero

### Module Classification
Automatically classify parameters as:
- `embedding`: Word embedding layer
- `q_proj`: Query projection layer
- `k_proj`: Key projection layer
- `v_proj`: Value projection layer
- `o_proj`: Output projection layer
- `moe`: Mixture of experts layer
- `norm`: Normalization layer
- `output`: Output layer

### Clustering Analysis
- Use elbow method to determine optimal number of clusters
- K-means clustering algorithm
- Result visualization

## 🔍 Usage Examples

### Analyze Qwen3 model (recommended)
```bash
# Use GPU acceleration to analyze Qwen3 model
python scripts/transformer_analysis.py \
    --model_path data/models \
    --out data/outputs/qwen3_analysis.csv \
    --model_id "Qwen/Qwen3-4B" \
    --rank_k 512
```

### Analyze Llama-2 model
```bash
# Download model
python scripts/download.py --model_id meta-llama/Llama-2-7b-hf --out data/models/

# Use GPU acceleration analysis
python scripts/transformer_analysis.py --model_path data/models --out data/outputs/llama2_analysis.csv
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Insufficient**
   ```
   ⚠️  GPU OOM, falling back to CPU for matrix (4096, 4096)
   ```
   **Solution**: 
   - Reduce `--rank_k` parameter value
   - Close other GPU applications
   - Use `--no_gpu` to force CPU mode

2. **CUDA Not Available**
   ```
   ❌ CUDA not available
   ```
   **Solution**:
   - Check NVIDIA driver installation
   - Verify PyTorch CUDA version
   - Confirm GPU hardware support

3. **Memory Insufficient**
   - Use `--low_cpu_mem_usage` parameter
   - Reduce precision (float16)
   - Process large models in batches

4. **Download Failure**
   - Check network connection
   - Use mirror sources
   - Retry download

5. **Unsupported Model Format**
   - Ensure model supports safetensors format
   - Check model structure

### Logging and Debugging
- All scripts output detailed log information
- Use `--verbose` parameter for more debug information

## 🤝 Contributing

Welcome contributions of code and ideas! Please follow these principles:

1. **Plan First, Execute Later**: Write a minimum viable plan before starting each task
2. **Minimal Necessary Abstraction**: Keep code concise, avoid over-engineering
3. **Reproducibility**: Fix random seeds, record version and environment information
4. **Documentation**: Add documentation and examples for new features

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Hugging Face team for providing the transformers library
- PyTorch team for providing the deep learning framework
- Open source community for support and contributions

## 📞 Contact

For questions or suggestions, please contact us through:
- Submit Issues
- Send emails
- Participate in discussions

---

**Note**: This project is for research and learning purposes only. Please comply with the relevant model usage terms and licenses.

