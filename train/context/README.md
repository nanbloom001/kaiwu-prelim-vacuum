# Train Context Folder

`train/context/` 保存的是“应该跟着仓库一起走”的高价值上下文：诊断、交接、优化记录、评估方案、服务器操作说明，以及少量结果摘要。

从 2026-04-16 起，这个目录改成了二级分类，避免所有文档都堆在根目录。

## 顶层保留文件

- `README.md`
  - 当前索引文件。
- `CHANGELOG.md`
  - 关键改动时间线，适合作为总入口之一。

## 二级目录说明

### `benchmark/`

benchmark 设计、使用方式、并行化交接。

关键文件：

- `benchmark/BENCHMARK_SYSTEM.md`
- `benchmark/BENCHMARK_PARALLEL_HANDOFF_20260416.md`
- `benchmark/BENCHMARK_TARGET_3C4R_20260425_NOTES.md`

### `data/`

结构化实验结果和 JSON 摘要。

当前包含：

- `data/DATAFETCH_BENCHMARK_RESULTS.json`
- `data/ENV_SCALING_RESULTS.json`
- `data/REPLAY_STABILITY_RESULTS.json`

### `diagnosis/`

问题诊断、瓶颈分析、失败原因归纳。

适合在排查训练异常、性能问题、reward 失效、电池死亡等问题时优先阅读。

### `handoff/`

交接文档和阶段性总结。

适合新接手的人快速建立上下文。

### `operations/`

服务器操作、监控、模型签名、同步等运维类文档。

### `optimization/`

训练吞吐、框架 patch、优化方案和优化结果。

### `sessions/`

按时间记录的 session 日志、阶段记录和训练过程备忘。

### `analysis/`

分析材料、提示词讨论，以及外部 AI 分析记录。

其中：

- `analysis/external_ai/` 保存 Gemini / GPT / Opus 之类外部分析稿

## 推荐阅读顺序

如果你第一次接手这个仓库，建议按这个顺序读：

1. `CHANGELOG.md`
2. `benchmark/BENCHMARK_SYSTEM.md`
3. `benchmark/BENCHMARK_PARALLEL_HANDOFF_20260416.md`
4. `handoff/HANDOFF_20260413.md`
5. `diagnosis/DIAGNOSIS_REMEDIATION_REPORT_20260409.md`
6. `optimization/FRAMEWORK_ACCELERATION_GUIDE.md`

## 什么文件应该放进来

- 稳定有价值的诊断文档
- 关键交接文档
- benchmark / experiment 方案说明
- 结构化小型结果文件
- 服务器和监控相关操作说明

## 什么文件不应该放进来

- 大日志目录
- benchmark 原始 step 日志
- checkpoint / pkl / zip 模型文件
- 容器导出的备份副本
- 临时截图、一次性命令输出

目标是让这个目录保持：

- 可提交
- 可导航
- 可直接交接
