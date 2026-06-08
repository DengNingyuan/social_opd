# Social-OPD Geo 实验方案

## 研究问题

这个版本验证的是：

> 不直接让模型 verbal self-report confidence，而是只让模型输出测量分数；
> confidence 由生成 token 的 Logit Geom. 计算。这样能否减少 verbal
> confidence 的主观偏差，同时保留 tolerance calibration 评估接口？

## 数据

当前 formality 数据保留：

- `text`
- `llm_predict_value`
- `ground_truth`

CSV 里若仍有 `llm_confidence` 列，geo 版预处理会忽略它。

训练 completion 只包含：

```json
{"score": 47}
```

prompt 不要求模型报告 confidence。

## Confidence

推理时使用 Logit Geom.：

```text
confidence = exp(mean(log p(generated_token_i)))
```

实现位置：

- `src/inference.py`
- `geometric_mean_confidence(token_logprobs)`

## Commercial API

`scripts/deepseek_three_aspects.py` 调用 `deepseek-chat`，要求商业 API 同时返回三类因素的文本描述和分数：

- `data_factors`
- `task_factors`
- `annotator_factors`

以及最终属性分数 `score`。

输出目录固定为：

```text
/home/nydeng/social_opd_all/social_opd_geo/data/commercial_api
```

## 指标

- Measurement: tolerance accuracy, Spearman.
- Calibration: tolerance ECE, Brier.
- Confidence source: Logit Geom., not verbal self-report.
