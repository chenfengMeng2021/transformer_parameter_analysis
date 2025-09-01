# AI Block Research

一个用于下载、读取和分析大语言模型参数的研究工具集。本项目通过下载与读取大模型参数，并用Jupyter Notebook进行结构化分析（embedding、各层矩阵维度与秩、统计量等）。

## 🚀 功能特性

- **模型下载**: 从Hugging Face自动下载模型权重文件
- **参数分析**: 分析模型参数的形状、秩、统计量等
- **嵌入分析**: 专门的embedding层分析，包括K-means聚类
- **Transformer分析**: 分析transformer各层的参数特性
- **数据导出**: 支持CSV、JSON等多种格式输出
- **可视化**: 生成分析图表和统计报告

## 📋 系统要求

- Python 3.10+
- 支持的操作系统: Linux, macOS, Windows
- 推荐内存: 8GB+ (取决于模型大小)

## 🛠️ 安装

### 1. 克隆项目
```bash
git clone <repository-url>
cd AI_Block_research
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
```

## 📁 项目结构

```
AI_Block_research/
├── data/                          # 数据目录
│   ├── models/                    # 模型缓存目录
│   └── outputs/                   # 分析结果输出
├── scripts/                       # 核心脚本
│   ├── download.py               # 模型下载脚本
│   ├── analyze.py                # 参数分析脚本
│   ├── embedding_analysis.py     # 嵌入层分析脚本
│   └── transformer_analysis.py   # Transformer分析脚本
├── notebooks/                     # Jupyter分析笔记本
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

### 3. 嵌入层分析
```bash
# 分析embedding层并进行聚类
python scripts/embedding_analysis.py --model_path data/models/Qwen/Qwen3-4B --out data/outputs/embedding_analysis.csv
```

## 📊 输出格式

### 参数分析输出 (CSV)
包含以下字段：
- `model_path`: 模型路径
- `model_id`: 模型标识符
- `revision`: 模型版本
- `param_name`: 参数名称
- `layer_index`: 层索引
- `block_type`: 模块类型 (embedding/q_proj/k_proj/v_proj/o_proj/moe/output)
- `shape`: 参数形状
- `rows`, `cols`: 行数和列数
- `rank`: 矩阵秩
- `mean`, `std`, `var`: 统计量
- `sparsity`: 稀疏度
- `dtype`: 数据类型
- `created_at`: 创建时间

### 嵌入分析输出
- 聚类结果
- 可视化图表
- 统计报告

## 🔧 脚本说明

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
- 肘部法则确定最优聚类数
- 可视化结果

**参数:**
- `--model_path`: 模型路径 (必需)
- `--out`: 输出文件路径 (必需)
- `--random_state`: 随机种子 (可选)

## 📈 分析特性

### 统计量计算
- **形状分析**: 自动识别参数维度
- **秩分析**: 计算矩阵的近似秩
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
- `output`: 输出层

### 聚类分析
- 使用肘部法则确定最优聚类数
- K-means聚类算法
- 结果可视化

## 🔍 使用示例

### 分析Llama-2模型
```bash
# 下载模型
python scripts/download.py --model_id meta-llama/Llama-2-7b-hf --out data/models/

# 分析参数
python scripts/analyze.py --model_path data/models/meta-llama/Llama-2-7b-hf --out data/outputs/llama2_params.csv

# 分析embedding
python scripts/embedding_analysis.py --model_path data/models/meta-llama/Llama-2-7b-hf --out data/outputs/llama2_embedding.csv
```

### 分析Qwen模型
```bash
# 下载模型
python scripts/download.py --model_id Qwen/Qwen3-4B --out data/models/

# 分析参数
python scripts/analyze.py --model_path data/models/Qwen/Qwen3-4B --out data/outputs/qwen_params.csv

# 分析embedding
python scripts/embedding_analysis.py --model_path data/models/Qwen/Qwen3-4B --out data/outputs/qwen_embedding.csv
```

## 📚 Jupyter Notebook分析

项目包含Jupyter notebook用于深入分析：

1. **00_setup.ipynb**: 环境设置和依赖检查
2. **10_summary_stats.ipynb**: 汇总统计和可视化
3. **20_per_layer.ipynb**: 按层分析参数特性
4. **30_embedding_output.ipynb**: 聚焦embedding和输出层

## 🐛 故障排除

### 常见问题

1. **内存不足**
   - 使用`--low_cpu_mem_usage`参数
   - 降低精度 (float16)
   - 分批处理大模型

2. **下载失败**
   - 检查网络连接
   - 使用镜像源
   - 重试下载

3. **模型格式不支持**
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
