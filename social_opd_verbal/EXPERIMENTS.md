# Social-OPD / MAUD 实验方案

## 研究问题

MAUD 要验证的不是“API 给出的 uncertainty 就是真值”，而是一个更稳妥的问题：

> 把单一 confidence 拆成有理论含义的诊断不确定性维度，并通过 OPD
> 蒸馏到部署学生模型后，是否能在保持测量准确性的同时提升校准一致性和可解释性？

## 数据

沿用 calibration paper 的 8 个 social measurement 数据集：

- `politeness`
- `formality`
- `hatespeech`
- `argument_quality`
- `humicroedit`
- `EmoBank_valence`
- `EmoBank_arousal`
- `EmoBank_dominance`

每条样本保留 `text`、原 LLM score/confidence、human score。MAUD 训练目标替换为：

```text
score* = human ground_truth
confidence* = diagnostic-context teacher confidence
u* = commercial API or bootstrap diagnostic uncertainty
```

其中 `u*` 包含三类不确定性：

- `data_context`: 文本或语境证据不足造成的不确定性。
- `task_construct`: 构念定义、量表边界或任务说明含混造成的不确定性。
- `annotator_perspective`: 合理标注者因价值、文化、视角不同而可能分歧造成的不确定性。

## 主实验

| system | output | teacher context | loss |
| --- | --- | --- | --- |
| zero-shot scalar | score + confidence | none | none |
| scalar SFT | score + confidence | none | CE + score/conf MSE |
| evidence teacher | score + confidence + evidence | evidence | CE + score/conf MSE |
| MAUD | score + confidence + 3 uncertainty dimensions | diagnostic context | CE + MSE + consistency + KL |

## Loss 设计理由

MAUD 的 loss 不是把几项随意相加，而是对应 social measurement 中的不同失败模式：

- `CE` 负责让模型稳定生成可解析 JSON。如果输出格式不稳定，后续 score、confidence 和 uncertainty 都无法可靠使用。
- `score MSE` 负责测量准确性，避免为了校准或解释性牺牲 human score 对齐。
- `confidence MSE` 负责保留下游最常用的 scalar confidence 接口。
- `uncertainty MSE` 负责把 confidence 背后的原因显式化，让模型区分“缺上下文”“构念边界不清”和“标注者视角分歧”。
- `consistency MSE` 负责约束 confidence 与 uncertainty 的关系，降低“高 confidence 但高 uncertainty”的自相矛盾输出。
- `KL` 负责 OPD 蒸馏：teacher 看到 privileged diagnostic context，student 只看到部署输入；KL 让 student 在无需额外上下文的情况下接近 teacher 的输出分布。

## 消融

| ablation | change | purpose |
| --- | --- | --- |
| `no_kl` | set `lambda_kl=0` | test OPD-style diagnostic-context distillation |
| `no_consistency` | set `lambda_consistency=0` | test confidence-uncertainty coupling |
| `scalar_only` | remove uncertainty MSE and KL | compare scalar calibration interface |
| `student_lite` | disable evidence in data prep | test numeric-only stability |
| `student_full` | include evidence strings | test interpretability output |

## 关键指标

- Measurement: MAE, RMSE, Spearman.
- Calibration: tolerance accuracy, ECE, Brier.
- Diagnostic consistency: Spearman(`1-confidence`, mean uncertainty), violation rate.
- Fidelity: confidence/uncertainty MSE to diagnostic labels.

## 推荐运行顺序

1. `scalar_only`: 确认普通 score + confidence SFT 的下限。
2. `no_kl`: 检查诊断 value loss 本身是否有效。
3. `no_consistency`: 检查 confidence 和 uncertainty 是否会自然对齐。
4. `maud_full`: 跑完整方法，比较准确性、校准和一致性三组指标。
5. `student_lite` vs `student_full`: 判断 evidence 文本是否提升解释性，或是否增加生成噪声。

## 推荐论文叙事

不要声称 API uncertainty 是真实不确定性标签。更稳妥的 claim 是：

> MAUD turns scalar confidence into a theory-informed diagnostic interface and
> shows that this interface can be distilled into a cheaper social measurement
> model with better internal consistency and interpretability.
