# Model Analysis Report

- Model ID: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

- Local Path: `/home/user/transformer_parameter_analysis/data/models/deepseek-ai-DeepSeek-R1-Distill-Llama-8B`

- Parameter Count: 8B (8,030,261,248)

- Vocab Size: 128256

- Layers: 32

- Hidden Size: 4096

- Attention Heads: 32

- RoPE: 500000.0


## Results

- Transformer CSV: `/home/user/transformer_parameter_analysis/data/outputs/deepseek-ai-DeepSeek-R1-Distill-Llama-8B/transformer_matrix_analysis.csv`

- Embedding silhouette (non-singleton clusters): -0.0109

- Embedding ANOVA Plot: `/home/user/transformer_parameter_analysis/data/outputs/deepseek-ai-DeepSeek-R1-Distill-Llama-8B/anova_curve.png`


### Embedding Clusters (summary)

| cluster_id | size | percentage | avg_distance | silhouette_score | meaningful_text |

| --- | --- | --- | --- | --- | --- |


## Notes

- CSV is sorted by numeric layer_index; non-layer rows come last.

- Ranks reported are full numerical ranks (no truncation).
