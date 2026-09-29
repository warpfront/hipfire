# hipfire decide vs Jev

Calibration applied: choice T=1, score T=1, noul T=1 (no calibration.json, so calibrated = raw). Calibrated = softmax(log p / T) per question type (spec §13.3); accuracy cannot change.

Not every jev-bench task or calibration set is committed to this repository: a source dataset's own licence terms have to clearly permit redistributing derived per-row data (label + served probabilities, no text) before its frozen rows are staged into git — see `bench/jev/DATA-LICENSES.md` for the per-dataset decision and sources. A row missing below either was not evaluated for this model, or is withheld for that reason.

## jev-bench (500 fixed-seed rows per task)

| task | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 500 | 0.782 | 0.546 | 0.099 | 0.188 | 0.188 | 2.716 | 2.716 |

## jev-ood-calibration

| set | type | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synth | choice | 300 | 0.890 | 0.897 | 0.084 | 0.055 | 0.055 | 0.441 | 0.441 |
| synth | score | 300 | 0.447 | 0.443 | 0.306 | 0.091 | 0.091 | 1.195 | 1.195 |
| synth | noul | 300 | 0.917 | 0.843 | 0.072 | 0.070 | 0.070 | 0.447 | 0.447 |

## score answers: mean |E[score] - gold|

| rows | n | raw | calibrated |
|---|---:|---:|---:|
| synth (score) | 300 | 0.699 | 0.699 |
