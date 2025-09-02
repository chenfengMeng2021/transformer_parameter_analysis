# Model Analysis Report

- Model ID: `bert-base-uncased`

- Local Path: `/home/user/transformer_parameter_analysis/data/models/bert-base-uncased`

- Parameter Count: 110M (110,106,428)

- Vocab Size: 30522

- Layers: 12

- Hidden Size: 768

- Attention Heads: 12


## Results

- Transformer CSV: `/home/user/transformer_parameter_analysis/data/outputs/bert_base_uncased/transformer_matrix_analysis.csv`

- Embedding silhouette (non-singleton clusters): 0.0625

- Embedding ANOVA Plot: `/home/user/transformer_parameter_analysis/data/outputs/bert_base_uncased/anova_curve.png`


### Embedding Clusters (summary)

| cluster_id | size | percentage | avg_distance | silhouette_score | meaningful_text |

| --- | --- | --- | --- | --- | --- |


## Notes

- CSV is sorted by numeric layer_index; non-layer rows come last.

- Ranks reported are full numerical ranks (no truncation).
