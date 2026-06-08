# Formality Self-OPD Idea

## Motivation

This experiment tests whether a very small prompt-side teacher intervention can improve calibration for LLM-based social science measurement.

The target task is Pavlick Formality. The model predicts:

- `score`: linguistic formality on a 0-100 scale.
- `confidence`: probability from 0 to 1 that the score is within +/-10 points of the human value.

The key question is simple: if the teacher uses a slightly more deliberate measurement prompt, can a student distilled from that teacher become better calibrated while preserving measurement quality?

## Dataset

The current first dataset is `formality`.

Dataset paths are configured in:

```text
configs/datasets.json
```

Current schema:

```text
text,llm_predict_value,llm_confidence,ground_truth
```

Future datasets should be added to the same JSON file with their own `train`, `test`, `name`, `attribute_name`, `attribute_definition`, and `teacher_instruction` fields. Each dataset should train its own model so calibration behavior is not mixed across constructs too early.

## Prompt Design

The student prompt follows the social science measurement prompt style from Appendix B.1 of *Assessing and Mitigating Miscalibration in LLM-Based Social Science Measurement* and the local `Calibration-dev-baselines` rating prompt. Prompt content is driven by each dataset's config entry, so datasets with the same CSV schema can use different construct definitions.

The formality definition is:

```text
Rate the linguistic formality of the text. Near 0 means extremely informal,
and near 100 means extremely formal, with intermediate values as appropriate.
Use your intuitive judgment of formality and use the full ratings scale.
```

The teacher prompt is `teacherv1`.

In `teacherv1`, the teacher is the same base model as the student, but the prompt additionally asks it to consider formality from three angles:

- data factors: ambiguity, missing context, noisy wording, or insufficient evidence in the text itself
- task factors: underspecified construct definition, scale interpretation, or boundary between adjacent score levels
- annotator factors: disagreement caused by annotator preferences, cultural background, social perspective, values, or interpretation style

The teacher still outputs only:

```json
{"score": 0, "confidence": 0.0}
```

It does not output the three angles. The three-angle instruction is only a latent measurement guide.

## Loss

The first loss version is `lossv1`.

`lossv1` is the classic Self-OPD / On-Policy Self-Distillation objective:

```text
loss = lambda_kl * KL(pi_student(. | student_prompt) || stopgrad(pi_teacher(. | teacher_prompt)))
```

This is the student-to-teacher Reverse KL. The teacher distribution is detached. Student and teacher share the same initial model family; they differ by prompt.

Loss descriptions are stored in:

```text
configs/losses.json
```

This makes it easy to add `lossv2`, `lossv3`, etc. later without losing track of what each version means.

## Experiments

There are two main scripts:

```bash
bash scripts/infer_formality_3b.sh
bash scripts/train_then_infer_formality_opd.sh
```

Direct inference evaluates the base 3B model without OPD training.

Train-then-infer trains one model per dataset using `lossv1` and `teacherv1`, then evaluates on the configured test split.

Output paths include:

```text
date
mode
loss version
dataset name
teacher version
model name
```

Each run writes:

```text
predictions.jsonl
metrics.json
run_metadata.json
```

Training runs also save:

```text
model/
```

## Code Organization

The repository keeps the standard Python `src/` layout:

```text
src/
```

Core extension points:

```text
preprocess.py  CSV to prompt JSONL preprocessing
data.py        processed JSONL dataset loading and collator
prompts.py     student/teacher prompt builders
losses.py      lossv1 and future loss variants
metrics.py     ECE, Brier, MH, parsing generated JSON
inference.py   shared inference loop
infer.py       direct inference entry
train_opd.py   train-then-infer entry
```

The main train/infer scripts first run preprocessing, then operate only on processed JSONL files:

```text
data/processed/{dataset}/train.jsonl
data/processed/{dataset}/test.jsonl
```

## Metrics

Evaluation is on the test set.

Metrics:

- `ECE ↓`: tolerance-based expected calibration error.
- `Brier ↓`: mean squared error between confidence and tolerance correctness.
- `MH ↑`: Spearman rank correlation between predicted score and human ground truth.
- `accuracy`: tolerance accuracy where `abs(pred - ground_truth) <= 10`.

The main goal is not just higher MH. The goal is better calibration under a measurement-validity framing.
