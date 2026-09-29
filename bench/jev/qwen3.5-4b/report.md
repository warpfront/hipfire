# hipfire decide vs Jev

Calibration applied: choice T=1.653, score T=1.48, noul T=1.363 (from calibration.json). Calibrated = softmax(log p / T) per question type (spec §13.3); accuracy cannot change.

Not every jev-bench task or calibration set is committed to this repository: a source dataset's own licence terms have to clearly permit redistributing derived per-row data (label + served probabilities, no text) before its frozen rows are staged into git — see `bench/jev/DATA-LICENSES.md` for the per-dataset decision and sources. A withheld source (ag-news, duplicates, yelp-stars, offensive, sentiment-it, hellaswag) is never in this table, even when this checkout has also frozen it locally. A row missing below either was not evaluated for this model, or is withheld for that reason.

## Fit (held-out rows only, build 237bb7bba, spec §13.4)

| type | n | T | 95% interval | held-out NLL raw | held-out NLL cal | shipped |
|---|---:|---:|---|---:|---:|---|
| choice | 2400 | 1.653 | 1.596-1.705 | 1.978 | 1.705 | yes |
| score | 300 | 1.480 | 1.218-1.791 | 1.235 | 1.206 | yes |
| noul | 900 | 1.363 | 1.178-1.621 | 0.396 | 0.384 | yes |

## jev-bench (500 fixed-seed rows per task)

| task | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 500 | 0.782 | 0.546 | 0.099 | 0.188 | 0.067 | 2.716 | 2.132 |

Rows for datasets whose licences restrict redistributing derived results are withheld; see ../DATA-LICENSES.md.

## jev-ood-calibration

| set | type | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synth | choice | 300 | 0.890 | 0.897 | 0.084 | 0.055 | 0.052 | 0.441 | 0.390 |
| synth | score | 300 | 0.447 | 0.443 | 0.306 | 0.091 | 0.125 | 1.195 | 1.182 |
| synth | noul | 300 | 0.917 | 0.843 | 0.072 | 0.070 | 0.071 | 0.447 | 0.428 |

Rows for datasets whose licences restrict redistributing derived results are withheld; see ../DATA-LICENSES.md.

## score answers: mean |E[score] - gold|

| rows | n | raw | calibrated |
|---|---:|---:|---:|
| synth (score) | 300 | 0.699 | 0.686 |
