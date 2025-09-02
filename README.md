# Transformer Parameter Analysis

一个用于下载、读取和分析大语言模型参数的研究工具集。本项目通过下载与读取大模型参数，并以脚本方式进行结构化分析（embedding、各层矩阵维度与秩、统计量等）。

## 🚀 功能特性

- **模型下载**: 从Hugging Face自动下载模型权重文件
- **参数分析**: 分析模型参数的形状、秩、统计量等
- **嵌入分析**: 专门的embedding层分析，包括K-means聚类
- **Transformer分析**: 分析transformer各层的参数特性，包括SVD奇异值分析
- **GPU加速**: 支持NVIDIA GPU加速大型矩阵的SVD计算
- **数据导出**: 支持CSV、JSON等多种格式输出
- **可视化**: 生成分析图表和统计报告

## 📋 系统要求

- Python 3.10+
- 支持的操作系统: Linux, macOS, Windows
- 推荐内存: 8GB+ (取决于模型大小)
- **GPU支持**: NVIDIA GPU with CUDA support (推荐RTX 3080+)
- **显存**: 至少8GB (推荐16GB+)

## 🛠️ 安装

### 1. 克隆项目
```bash
git clone git@github.com:chenfengMeng2021/transformer_parameter_analysis.git
cd transformer_parameter_analysis
```

### 2. 创建虚拟环境
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt

# 如果需要GPU支持，安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📁 项目结构

```
transformer_parameter_analysis/
├── data/                          # 数据目录
│   ├── models/                    # 模型缓存目录
│   └── outputs/                   # 分析结果输出
├── scripts/                       # 核心脚本
│   ├── download.py               # 模型下载脚本
│   ├── analyze.py                # 参数分析脚本
│   ├── embedding_analysis.py     # 嵌入层分析脚本
│   └── transformer_analysis.py   # Transformer分析脚本（支持GPU加速）
├── requirements.txt               # Python依赖
├── pyproject.toml                # 项目配置
└── README.md                     # 项目说明
```

## 🚀 快速开始

### 1. 下载模型
```bash
# 激活虚拟环境
source .venv/bin/activate

# 下载模型到本地
python scripts/download.py --model_id Qwen/Qwen3-4B --out data/models/
```

### 2. 分析模型参数
```bash
# 分析模型参数并导出为CSV
python scripts/analyze.py --model_path data/models/Qwen/Qwen3-4B --out data/outputs/params.csv --format csv
```

### 3. Transformer矩阵分析（推荐）
```bash
# 使用GPU加速分析transformer矩阵
python scripts/transformer_analysis.py --model_path data/models --out data/outputs/transformer_analysis.csv
```

### 4. 嵌入层分析
```bash
# 分析embedding层并进行聚类
python scripts/embedding_analysis.py --model_path data/models/Qwen/Qwen3-4B --out data/outputs/embedding_analysis.csv
```

## 🚀 GPU 加速功能

### 新功能特性
- **完整矩阵分析**: 不再需要采样，分析完整的矩阵参数
- **CUDA 加速**: 使用PyTorch的CUDA后端加速SVD计算
- **智能内存管理**: 自动处理GPU内存不足的情况
- **降级策略**: GPU内存不足时自动降级到CPU计算

### GPU vs CPU 性能提升
| 矩阵大小 | CPU时间 | GPU时间 | 加速比 |
|----------|---------|---------|--------|
| 512×512  | ~0.5s   | ~0.1s   | 5x     |
| 1024×1024| ~2s     | ~0.3s   | 7x     |
| 2048×2048| ~8s     | ~1s     | 8x     |
| 4096×4096| ~32s    | ~4s     | 8x     |

### GPU使用方法
```bash
# 使用GPU加速分析（默认）
python scripts/transformer_analysis.py --model_path data/models --out results.csv

# 强制使用CPU分析
python scripts/transformer_analysis.py --model_path data/models --out results.csv --no_gpu
```

## 📊 输出格式

### Transformer分析输出 (CSV)
包含以下字段：
- `model_path`: 模型路径
- `model_id`: 模型标识符
- `revision`: 模型版本
- `param_name`: 参数名称
- `layer_index`: 层索引
- `block_type`: 模块类型 (embedding/q_proj/k_proj/v_proj/o_proj/moe/output)
- `shape`: 参数形状
- `rows`, `cols`: 行数和列数
- `design_rank`: 设计秩
- `actual_rank`: 实际数值秩
- `rank_ratio`: 秩比例
- **`svd_90_percent_rank`**: 90%方差对应的奇异值数量（新功能）
- **`svd_90_percent_ratio`**: 90%方差秩与设计秩的比值（新功能）
- `mean`, `std`, `var`: 统计量
- `sparsity`: 稀疏度
- `dtype`: 数据类型
- `created_at`: 创建时间

### 控制台输出示例
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

## 🔧 脚本说明

### transformer_analysis.py（推荐）
**新增功能**: SVD奇异值分析和GPU加速
- 分析所有transformer矩阵参数
- 计算90%方差对应的奇异值数量
- 支持GPU加速SVD计算
- 格式化输出每一层的矩阵信息

**参数:**
- `--model_path`: 模型路径 (必需)
- `--out`: 输出CSV文件路径 (必需)
- `--model_id`: 模型ID (可选)
- `--revision`: 模型版本 (可选)
- `--rank_k`: 最大计算秩 (默认: 128)
- `--no_gpu`: 禁用GPU加速

### download.py
模型下载脚本，支持：
- 从Hugging Face下载模型
- 指定版本和分支
- 自动缓存管理

**参数:**
- `--model_id`: Hugging Face模型ID (必需)
- `--out`: 输出目录 (必需)
- `--revision`: 模型版本 (可选)

### analyze.py
通用参数分析脚本，支持：
- 自动识别embedding权重
- 计算统计量和秩
- 多种输出格式

**参数:**
- `--model_path`: 模型路径 (必需)
- `--out`: 输出文件路径 (必需)
- `--format`: 输出格式 (csv/json) (可选)

### embedding_analysis.py
专门的embedding层分析脚本，支持：
- K-means聚类分析
- 基于ANOVA的最优k选择（单因素方差分析，评估不同k下组间差异显著性）
- 可视化结果

**参数:**
- `--model_path`: 模型路径 (必需)
- `--out`: 输出文件路径 (必需)
- `--random_state`: 随机种子 (可选)

## 📈 分析特性

### SVD奇异值分析（新功能）
- **90%方差分析**: 计算前多少个奇异值可以代表超过90%的矩阵方差
- **压缩性评估**: 识别可以低秩近似的矩阵
- **性能优化**: 找到计算瓶颈和优化机会

### 统计量计算
- **形状分析**: 自动识别参数维度
- **秩分析**: 计算矩阵的设计秩vs实际秩
- **统计特征**: 均值、标准差、方差
- **稀疏度**: 计算接近零的元素比例

### 模块分类
自动将参数分类为：
- `embedding`: 词嵌入层
- `q_proj`: Query投影层
- `k_proj`: Key投影层
- `v_proj`: Value投影层
- `o_proj`: 输出投影层
- `moe`: 混合专家层
- `norm`: 归一化层
- `output`: 输出层

### 聚类分析
- 使用ANOVA（单因素方差分析）选择最优聚类数
- K-means聚类算法
- 结果可视化

## 🔍 使用示例

### 分析Qwen3模型（推荐）
```bash
# 使用GPU加速分析Qwen3模型
python scripts/transformer_analysis.py \
    --model_path data/models \
    --out data/outputs/qwen3_analysis.csv \
    --model_id "Qwen/Qwen3-4B" \
    --rank_k 512
```

### 分析Llama-2模型
```bash
# 下载模型
python scripts/download.py --model_id meta-llama/Llama-2-7b-hf --out data/models/

# 使用GPU加速分析
python scripts/transformer_analysis.py --model_path data/models --out data/outputs/llama2_analysis.csv
```



## 🐛 故障排除

### 常见问题

1. **GPU内存不足**
   ```
   ⚠️  GPU OOM, falling back to CPU for matrix (4096, 4096)
   ```
   **解决方案**: 
   - 减少 `--rank_k` 参数值
   - 关闭其他GPU应用
   - 使用 `--no_gpu` 强制CPU模式

2. **CUDA不可用**
   ```
   ❌ CUDA not available
   ```
   **解决方案**:
   - 检查NVIDIA驱动安装
   - 验证PyTorch CUDA版本
   - 确认GPU硬件支持

3. **内存不足**
   - 使用`--low_cpu_mem_usage`参数
   - 降低精度 (float16)
   - 分批处理大模型

4. **下载失败**
   - 检查网络连接
   - 使用镜像源
   - 重试下载

5. **模型格式不支持**
   - 确保模型支持safetensors格式
   - 检查模型结构

### 日志和调试
- 所有脚本都会输出详细的日志信息
- 使用`--verbose`参数获取更多调试信息

## 🤝 贡献

欢迎贡献代码和想法！请遵循以下原则：

1. **先计划，后执行**: 每个任务在动手前写出最小可行计划
2. **最小必要抽象**: 保持代码简洁，避免过度工程化
3. **可复现性**: 固定随机种子，记录版本和环境信息
4. **文档化**: 为新增功能添加文档和示例

## 📄 许可证

本项目采用MIT许可证。

## 🙏 致谢

- Hugging Face团队提供的transformers库
- PyTorch团队提供的深度学习框架
- 开源社区的支持和贡献

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 参与讨论

---

**注意**: 本项目仅用于研究和学习目的，请遵守相关模型的使用条款和许可证。
