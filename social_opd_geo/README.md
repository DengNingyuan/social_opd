# Social-OPD / MAUD

This repository contains a runnable scaffold for **Measurement-Aware
Uncertainty Distillation (MAUD)**, an on-policy distillation method for
calibrated social-science measurement.

## Why Social Measurement Needs More Than Scalar Confidence

In social-science measurement, errors often come from several different sources:
the text may be under-specified, the construct definition may be fuzzy, or
reasonable annotators may disagree. A single confidence number collapses all of
those cases into one value, so it is hard to tell whether the model is uncertain
because it lacks context, because the rubric is ambiguous, or because the item is
genuinely perspective-dependent.

MAUD keeps scalar confidence, but makes it accountable to a diagnostic
uncertainty interface. The student reports:

- `score`: construct measurement score, supervised by human labels.
- `confidence`: overall reliability of the measurement, computed at inference time with Logit Geom.
- `uncertainty_scores`: diagnostic uncertainty over three dimensions:
  `data_context`, `task_construct`, and `annotator_perspective`.
- optional `uncertainty_evidence`: short textual explanations for each
  dimension.

The student sees only the cheap deployment input:

```text
construct definition + rating scale + uncertainty dimension definitions + text
```

The teacher is the same model family conditioned on privileged diagnostic
context:

```text
student: pi_theta(. | x)
teacher: pi_theta(. | x, z_diag)
```

where `z_diag` can come from a commercial API, human annotation notes, or the
included deterministic bootstrap labels for dry runs.

## Training Objective

The loss combines language modeling, numeric measurement supervision, diagnostic
consistency, and on-policy distillation:

```text
L = lambda_ce * CE(y_tilde)
  + lambda_score_mse * MSE(score_hat, score*)
  + lambda_conf_mse * MSE(conf_hat, confidence*)
  + lambda_unc_mse * MSE(u_hat, u*)
  + lambda_consistency * MSE(conf_hat, 1 - mean(u_hat))
  + lambda_kl * KL(pi_student || stopgrad(pi_teacher))
```

Each term has a different job:

- `CE(y_tilde)` trains the model to emit strict structured JSON. Without this
  term, the numeric heads may learn useful values while the generated output is
  brittle or unparsable.
- `MSE(score_hat, score*)` anchors the measurement score to the human label. The
  method should improve calibration without drifting away from the construct
  being measured.
- `MSE(conf_hat, confidence*)` distills the teacher/API confidence target. This
  preserves the ordinary scalar confidence interface needed by downstream users.
- `MSE(u_hat, u*)` teaches the student to expose why confidence is low or high
  along the three social-measurement uncertainty dimensions.
- `MSE(conf_hat, 1 - mean(u_hat))` prevents an inconsistent output such as high
  confidence with high diagnostic uncertainty. It is intentionally soft: the
  model can still learn exceptions from data, but the default geometry ties
  confidence to the diagnostic interface.
- `KL(pi_student || stopgrad(pi_teacher))` is the OPD-style distillation term.
  The same model is run once with deployment context and once with privileged
  diagnostic context. The teacher distribution is detached, so the student is
  nudged toward diagnostic-context behavior without requiring that diagnostic
  context at deployment time.

The design goal is not to claim that API uncertainty is ground truth. The more
careful claim is that diagnostic uncertainty labels provide a structured
training signal that can make a cheaper student more internally consistent and
more interpretable than scalar confidence alone.

## Experiment Plan

Main baselines:

- `zs_scalar`: zero-shot or SFT student with score + scalar confidence only.
- `direct_teacher`: teacher produces score directly without evidence; confidence comes from Logit Geom.
- `evidence_teacher`: teacher reasons with evidence then produces score; confidence comes from Logit Geom.
- `maud`: evidence + three uncertainty dimensions + consistency + DCOD KL.

Core ablations:

- `no_kl`: remove diagnostic-context KL.
- `no_consistency`: remove confidence-uncertainty consistency.
- `no_uncertainty`: remove uncertainty dimensions.
- `student_lite`: numeric structured output only.
- `student_full`: numeric output + evidence text.
- `fewshot_teacher`: optional, diagnostic-context teacher with exemplars.

Metrics:

- Measurement: MAE, RMSE, Spearman.
- Calibration: tolerance ECE, Brier.
- Diagnostic consistency: correlation between `1 - confidence` and
  average uncertainty; violation rate for high confidence with high uncertainty.
- Distillation fidelity: MSE to teacher/API targets for confidence and
  uncertainty vectors.

## Quick Start

Install:

```bash
pip install -e '.[dev]'
```

Run v1 OPD training followed by inference:

```bash
scripts/train_then_infer_formality_opd.sh
```

Run direct Qwen2.5-3B inference with the v1 prompt:

```bash
scripts/infer_formality_3b.sh
```

Evaluate a prediction JSONL:

```bash
python -m evaluate \
  --pred_file outputs/predictions.jsonl \
  --out_file outputs/metrics.json
```

Run tests:

```bash
python -m pytest
```

For commercial API labels, provide secrets only through environment variables,
for example `OPENAI_API_KEY`; this code never prints key values.

## Environment Note

If `transformers` fails with `cannot import name 'DEFAULT_CIPHERS' from
urllib3.util.ssl_`, the local `boto3/botocore` stack is incompatible with
`urllib3>=2`. Use a clean environment or pin urllib3:

```bash
pip install 'urllib3<2'
```

Alternatively upgrade the AWS stack:

```bash
pip install -U boto3 botocore
```
