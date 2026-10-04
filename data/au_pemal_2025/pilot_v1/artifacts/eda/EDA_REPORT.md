# AU-PEMal pilot EDA

## Dataset overview

- Candidate samples: 100
- Dynamic samples: 40
- Sandbox reports: 231
- Canonical events and embeddings: 11161
- Unique canonical event texts: 8234
- Canonical traces with known ordering: 0

## Dynamic sample class distribution

| Value | Count |
|---|---:|
| Benign | 20 |
| Malicious | 20 |

## Malware category distribution

| Value | Count |
|---|---:|
| Stealer | 5 |
| RAT | 5 |
| Trojan | 5 |
| Ransomware | 5 |

## Dynamic sample split

| Value | Count |
|---|---:|
| train | 26 |
| test | 9 |
| validation | 5 |

## Event types

| Value | Count |
|---|---:|
| file | 6015 |
| registry | 3656 |
| module | 787 |
| network | 280 |
| process | 249 |
| service | 98 |
| command | 55 |
| api | 21 |

## Event-count distribution per sample

Min 6, mean 279.02, median 240.5, max 1362.

## Embedding checks

All 11161 vectors are finite. L2 norm: min 1.000000, mean 1.000000, max 1.000000.

## Interpretation warning

All canonical traces have `ordering_known=0`. UMAP visualizes semantic event similarity only; neither event indices nor UMAP axes represent execution time. Sample-level plots use mean pooling over each sample's event embeddings.
