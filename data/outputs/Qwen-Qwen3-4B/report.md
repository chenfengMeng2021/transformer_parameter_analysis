# Model Analysis Report

- Model ID: `Qwen/Qwen3-4B`

- Local Path: `/home/user/transformer_parameter_analysis/data/models/Qwen-Qwen3-4B`

- Parameter Count: 4B (4,022,468,096)

- Vocab Size: 151936

- Layers: 36

- Hidden Size: 2560

- Attention Heads: 32

- RoPE: 1000000


## Results

- Transformer CSV: `/home/user/transformer_parameter_analysis/data/outputs/transformer_matrix_analysis.csv`

- Embedding silhouette (non-singleton clusters): -0.0178

- Embedding ANOVA Plot: `/home/user/transformer_parameter_analysis/data/outputs/anova_curve.png`


### Embedding Clusters (summary)

| cluster_id | size | percentage | avg_distance | silhouette_score | meaningful_text |

| --- | --- | --- | --- | --- | --- |


## Notes

- CSV is sorted by numeric layer_index; non-layer rows come last.

- Ranks reported are full numerical ranks (no truncation).
