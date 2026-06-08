最简单 formality Self-OPD 版本

目的：
这一步只测一个很小的假设：单纯把 teacher prompt 多加“三个角度分析”
的引导，能不能通过 Self-OPD reverse KL 把 student 的 formality 打分和置信度拉得更好。

执行：

  cd /Users/lilyd/Documents/codex_lily/papers/opd/social_opd

直接推理 3B 模型，在 test 上计算指标：

  DATASET=formality \
  MODEL=Qwen/Qwen2.5-3B-Instruct \
  bash scripts/infer_formality_3b.sh

先训练，再在 test 上推理计算指标：

  DATASET=formality \
  MODEL=Qwen/Qwen2.5-3B-Instruct \
  bash scripts/train_then_infer_formality_opd.sh

两个脚本都会先运行 preprocess：

  PYTHONPATH=src python -m preprocess \
    --dataset_name formality \
    --output_dir data/processed/formality \
    --tolerance 10

后续训练和推理都只读取处理后的 JSONL：

  data/processed/formality/train.jsonl
  data/processed/formality/test.jsonl

双 L20 显存设置：

  DEVICE=auto 会把 student 放到 cuda:0
  TEACHER_DEVICE=auto 会在检测到第二张 CUDA 卡时把 teacher 放到 cuda:1

也可以显式指定：

  DEVICE=cuda:0 \
  TEACHER_DEVICE=cuda:1 \
  bash scripts/train_then_infer_formality_opd.sh

默认数据：

  train: data/formality_train_bert.csv
  test:  data/formality_test_bert.csv

数据集配置统一放在：

  configs/datasets.json

目前只用 formality。之后扩展新数据集时，在 datasets.json 里加一个名字，
sh 里改 DATASET 即可。每个 DATASET 会单独训练一个模型。
所有数据集使用同一个 CSV schema：

  text,llm_predict_value,ground_truth

每个 dataset 配置还需要写 prompt 字段：

  attribute_name
  attribute_definition
  teacher_instruction

训练和推理都会从这些字段生成 student/teacher prompt，不再硬编码 formality。

loss 配置：

  configs/losses.json

现在是 lossv1，后续新增 loss2/loss3 时只需要在 JSON 里写清楚 description，
并在代码里切换具体 loss 实现。

teacher 配置：

  configs/teachers.json

现在是 teacherv1：teacher 和 student 用同一个 3B 模型，只是 teacher prompt
多了三个维度的考虑说明，输出仍然只有 score。

默认模型：

  Qwen/Qwen2.5-3B-Instruct

原因：
你是 40G 显存，之前 7B 全参 GRPO 失败过；这里先按你的要求换到 3B，
但仍然不用 7B。3B 全参 + frozen teacher + batch_size=1 + grad_accum=16
+ bf16/gradient checkpointing + Adafactor，比 AdamW 更省显存，适合先验证这个最小想法。

输出：

直接推理：

  outputs/infer/{date}_infer_{loss_version}_{dataset_name}_{teacher_version}_{model}/

先训练再推理：

  outputs/train/{date}_train_{loss_version}_{dataset_name}_{teacher_version}_{model}/

每个输出目录包含：

  model/                  训练后的模型，仅训练脚本有
  predictions.jsonl
  metrics.json
  run_metadata.json

日志：

  logs/{date}_{mode}_{loss_version}_{dataset_name}_{teacher_version}_{model}.log

metrics.json 里会直接返回：

  accuracy
  ece
  brier
  mh

实现位置：

  preprocess: src/preprocess.py
  data:   src/data.py
  loss:   src/losses.py
  metric: src/metrics.py
  infer:  src/infer.py
  prompt: src/prompts.py 的 build_formality_opd_prompt
  train:  src/train_opd.py

当前 loss：

  lossv1 = Self-OPD / On-Policy Self-Distillation 的核心 Reverse KL

  total = lambda_kl * KL(student_prompt_distribution || stopgrad(teacher_prompt_distribution))

这里 teacher 和 student 不是两个不同模型；它们只是 prompt 不同。
teacher prompt 比 student prompt 多一段 instruction：
在给出最终 score 前，先从 disagreement-aware 的三个来源展开分析：
data factors、task factors、annotator factors。最终只输出 score JSON。
confidence 不由 prompt 直接询问，而是在 inference 时用 Logit Geom. 从生成 token logprobs 计算。

指标口径：

  ECE: tolerance-based ECE，correct = abs(pred - ground_truth) <= 10
  Brier: mean((confidence - correct)^2)
  MH: Spearman rank correlation between predicted score and human ground truth
