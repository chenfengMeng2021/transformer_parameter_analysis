# Model Analysis Report

- Model ID: `bert-base-uncased`

- Local Path: `/home/user/transformer_parameter_analysis/data/models/bert-base-uncased`

- Parameter Count: 110M (110,106,428)

- Vocab Size: 30522

- Layers: 12

- Hidden Size: 768

- Attention Heads: 12


## Results

- Transformer CSV: `/home/user/transformer_parameter_analysis/data/outputs/bert-base-uncased/transformer_matrix_analysis.csv`

- Embedding silhouette (non-singleton clusters): 0.0625

- Embedding ANOVA Plot: `/home/user/transformer_parameter_analysis/data/outputs/bert-base-uncased/anova_curve.png`


### Embedding Clusters (summary)

| cluster_id | size | percentage | avg_distance | silhouette_score | meaningful_text |

| --- | --- | --- | --- | --- | --- |

| 0 | 100 | 19.53 | 20.8092 | 0.0706 | [unused83] [unused55] [unused84] [unused82] [unused81] [unused56] [unused54] [un |

| 1 | 104 | 20.31 | 21.0118 | 0.0604 | [unused39] [unused12] [unused38] [unused11] [unused40] [unused13] [unused37] [un |

| 2 | 121 | 23.63 | 21.5423 | 0.0573 | [unused73] [unused72] [unused74] [unused75] [unused71] [unused77] [unused99] [un |

| 3 | 98 | 19.14 | 21.4022 | 0.0642 | [unused31] [unused30] [unused18] [unused32] [unused17] [unused45] [unused16] [un |

| 4 | 89 | 17.38 | 20.8368 | 0.0613 | [unused62] [unused63] [unused91] [unused90] [unused61] [unused92] [unused34] [un |


## Notes

- CSV is sorted by numeric layer_index; non-layer rows come last.

- Ranks reported are full numerical ranks (no truncation).
