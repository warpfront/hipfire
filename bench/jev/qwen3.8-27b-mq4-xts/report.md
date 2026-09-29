# hipfire decide vs Jev

Calibration applied: choice T=1, score T=1, noul T=1 (no calibration.json, so calibrated = raw). Calibrated = softmax(log p / T) per question type (spec §13.3); accuracy cannot change.

Not every jev-bench task or calibration set is committed to this repository: a source dataset's own licence terms have to clearly permit redistributing derived per-row data (label + served probabilities, no text) before its frozen rows are staged into git — see `bench/jev/DATA-LICENSES.md` for the per-dataset decision and sources. A row missing below either was not evaluated for this model, or is withheld for that reason.

## jev-bench (500 fixed-seed rows per task)

| task | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 500 | 0.782 | 0.718 | 0.099 | 0.138 | 0.138 | 1.418 | 1.418 |
| clinc150 | 500 | 0.914 | 0.888 | 0.028 | 0.044 | 0.044 | 0.481 | 0.481 |
| doc-yesno | 500 | 0.930 | 0.886 | 0.026 | 0.028 | 0.028 | 0.317 | 0.317 |
| ledgar | 500 | 0.740 | 0.752 | 0.138 | 0.121 | 0.121 | 1.209 | 1.209 |
| massive-en | 500 | 0.844 | 0.838 | 0.061 | 0.069 | 0.069 | 0.841 | 0.841 |
| massive-it | 500 | 0.836 | 0.810 | 0.057 | 0.089 | 0.089 | 0.989 | 0.989 |
| sms-spam | 500 | 0.986 | 0.972 | 0.058 | 0.014 | 0.014 | 0.091 | 0.091 |

## jev-ood-calibration

| set | type | n | Jev acc | hipfire acc | Jev ECE | ECE raw | ECE cal | NLL raw | NLL cal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| commonsense_qa | choice | 1221 | 0.881 | 0.849 | 0.030 | 0.049 | 0.049 | 0.492 | 0.492 |
| openbookqa | choice | 500 | 0.942 | 0.908 | 0.028 | 0.015 | 0.015 | 0.266 | 0.266 |
| synth | choice | 300 | 0.890 | 0.953 | 0.084 | 0.046 | 0.046 | 0.387 | 0.387 |
| synth | score | 300 | 0.447 | 0.547 | 0.306 | 0.134 | 0.134 | 1.007 | 1.007 |
| synth | noul | 300 | 0.917 | 0.917 | 0.072 | 0.037 | 0.037 | 0.286 | 0.286 |

## score answers: mean |E[score] - gold|

| rows | n | raw | calibrated |
|---|---:|---:|---:|
| synth (score) | 300 | 0.547 | 0.547 |
