# 闭环训练、固定留出地图 Benchmark 与数据驱动优化方案

## TL;DR
> **摘要**：为 `win_YJY` 分支建立一个保守、可回退但持续推进的自动实验闭环：前置参考最新 `linux-LTSPPO-charge-constraint` 分支构建本地 benchmark，再固定当前状态与 baseline，随后按“纯正面收益 → 低风险 → 奖励新增/改写 → 小幅重构 → 必要时网络改动”的优先级逐步优化，训练、评估、分析、决定保留或回退，直到 benchmark 平均分 **>900** 才停止。  
> **交付物**：
> - 对 `linux-LTSPPO-charge-constraint` benchmark/日志方案的本地适配说明。
> - 固定留出地图 `[4,7]` benchmark runner 与分析报告。
> - 可供 AI 深度分析的结构化日志：episode、step/action、planner、reward、battery/charger、death replay、per-map 指标。
> - baseline、迭代报告、死亡回放分析、最终 checkpoint 选择报告。
> - handoff 推荐的两个首批低风险修复：失败局不能晋升 best；coverage target 返航缓冲；后续可基于数据进入奖励设计、小幅重构或网络改动。
> - 每个接受的改动单独 commit，每个拒绝的改动通过 git 回退。
> **工作量**：Large  
> **可并行**：YES - 7 个执行波次  
> **关键路径**：T0 参考 LTSPPO benchmark → T1 固定 baseline → T2 补 benchmark 基础设施 → T3 跑 baseline benchmark → T4/T5 首批修复 → T8 深度分析 → T6 持续训练/评估/优化闭环 → T9 最终报告

## Context

### 原始请求
用户要求先阅读 `train/context/HANDOFF_20260425_WIN_YJY.md`，然后生成一个能“自行训练、自行看效果、自行运行 benchmark 深度分析、自行设计优化方案然后应用”的闭环方案。目标是在未用于训练的地图上达到 `2*10` 轮平均分 `>=900`，并且闭环必须持续到 benchmark 平均分 **>900** 才停止。事实约束：模型 20 分钟基本收敛，50 分钟左右完全收敛。边界：每次修改提交 git 便于回退；不是绝对不能改网络，但优化优先级为“纯正面收益 > 低风险 > 奖励新增/改写 > 小幅重构 > 必要时网络改动”；需要 changelog；可补 benchmark、死亡回放等基础设施；必须数据驱动且使用统一比较标准；不需要用户额外确认权限。

### 已确认需求
- 训练地图固定为 `[1,2,3,5,6,8,9,10]`。
- 留出/benchmark 地图固定为 `[4,7]`。
- 固定环境条件：`battery_max=150`、`max_step=1000`、`charger_count=3`、`robot_count=4`。
- 快速验收门槛：地图 `[4,7]` 各 `10` 局，共 `20` 局，平均 clean score **>900**；未超过 900 时闭环不得停止，只能升级优化策略或回退无效改动后继续。
- handoff 中的更高目标：最终尽量推进到 `>=950`，并保持低电池死亡率/碰撞率。
- 每个行为改动必须有独立 commit，失败则 git 回退。
- 新增前置要求：仿照最新远端分支 `linux-LTSPPO-charge-constraint` 构建 benchmark，但本地 benchmark 固定为 `1000` 步、最大电量 `150`、`4` 机器人、`3` 充电桩、`2*10` 次；重点是产出可供 AI 详细分析的日志。

### Metis 审查结论（已吸收）
- 必须先实现并提交 benchmark 基础设施，再做算法修复，否则无法形成干净 baseline。
- 当前已有 dirty 文件，不能盲目 reset；先记录初始状态。
- 当前已有 dirty 文件，因此 T1 的 baseline anchor 是硬门槛：未记录 branch/commit/diff/checkpoint hash/模型 mtime 前，不得开始 T2 之外的任何 benchmark 或算法修改。
- `[4,7]` 只能作为 benchmark，不得混入训练 sampler。
- `2*10` 样本偏少；接近门槛或波动大时必须追加更大样本确认。
- “正收益改动”必须量化为接受/拒绝门槛。

### Oracle 审查结论（已吸收）
- 这个系统必须是“保守实验 runner”，不是无约束自由优化器；但在连续证据表明低风险策略不足时，允许按升级阶梯进入奖励设计、小幅重构或网络改动。
- 每次只做一个小的、可逆的改动。
- 平均分不够；必须同时追踪最低分、方差、completed rate、battery fail rate、collision fail rate、clean_per_step、死亡回放。
- 网络结构改动、激进 reward 重写、checkpoint 删除或 cherry-pick 模型不能作为早期默认动作；其中网络/reward 重写只有在 benchmark 与日志证据支撑、且低优先级策略已不足时才能进入计划内迭代。

## Work Objectives

### 核心目标
建立一个可执行、可回退、证据驱动的闭环，使 `win_YJY` 分支在固定留出地图 `[4,7]` 上持续优化，直到 `2*10` benchmark 平均分 **>900**。默认先不改网络，但如果纯正面收益、低风险修复、奖励设计和小幅重构均无法达标，则允许在严格验证与回退保护下规划小范围网络改动。

### 交付物
- `train/context/BASELINE_YYYYMMDD_win_YJY.md`：当前状态与回退锚点。
- `train/run_holdout_benchmark.py`：固定 `[4,7]` benchmark runner。
- `train/analyze_holdout_benchmark.py`：benchmark + archive + death replay 分析器。
- `train/context/LTSPPO_BENCHMARK_ADAPTATION_YYYYMMDD.md`：记录从 `linux-LTSPPO-charge-constraint` 借鉴的 benchmark/日志设计与本地差异。
- `train/context/HOLDOUT_BENCHMARK_*.json` 与 `HOLDOUT_ANALYSIS_*.md`：小型摘要结果。
- `train_workflow.py` 中“失败局不能晋升 best checkpoint”的修复。
- `algorithm.py` 中 `COVERAGE_RETURN_BUFFER = 8.0` 的 coverage target 返航缓冲修复。
- `train/run_closed_loop_iteration.py` 或中文 runbook：训练→评估→分析→接受/回退闭环。
- 优化升级阶梯与自动决策规则：纯正面收益、低风险修复、奖励新增/改写、小幅重构、必要网络改动。
- 可扩展日志/分析工具：死亡回放、per-map、per-failure、reward component、路线/电量预算、checkpoint ranking、训练趋势对比。
- `train/context/CHANGELOG.md` 与 `LOG_YYYYMMDD_closed_loop_holdout.md`。
- `train/context/FINAL_SELECTION_YYYYMMDD.md`：最终 checkpoint 选择与提交准备报告。

### Definition of Done
- `python -m py_compile code/agent_ppo/algorithm/algorithm.py code/agent_ppo/feature/preprocessor.py code/agent_ppo/workflow/train_workflow.py train/run_holdout_benchmark.py train/analyze_holdout_benchmark.py` 通过。
- `python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 10 --dry-run` 能验证配置且不修改模型文件。
- benchmark 输出包含 map 4、map 7、combined 三组指标。
- benchmark 产出 AI 可分析日志，至少包含 episode summary、step/action trace 摘要、planner target/mode、reward components、battery/charger/slack、death replay 引用、模型文件 mutation check。
- 训练地图配置仍排除 `4` 和 `7`。
- benchmark runner 必须是 inference-only：不得走会触发 `latest/resume/best` 保存逻辑的训练 workflow 路径；如只能复用训练入口，必须显式禁用保存并用 mtime/hash 断言模型未变。
- 每个接受的改动有独立 commit；每个拒绝的改动有 git revert 记录。
- 快速门槛：`[4,7]` 共 `20` 局平均分 `>900`；未超过 900 时闭环继续。

### Must Have
- 先 baseline，后改算法。
- 固定训练/留出地图隔离。
- 一次一个可归因行为改动；奖励新增/改写、小幅重构、网络改动都必须拆成可独立验证的小步。
- 数值化接受/拒绝门槛。
- 每次实现/bug 排查后更新 changelog。
- 回退不得抹掉已有 dirty 状态。

### Must NOT Have
- 不把网络结构、tensor shape、模型层数或大规模 PPO optimizer 逻辑作为早期默认动作；只有在数据证明低风险/奖励/小幅重构不足时，才允许进入单独计划、单独 commit、单独 benchmark 的网络改动。
- 不整体 merge `origin/linux`。
- 不在训练中使用 `[4,7]`。
- 不提交 `train/log/`、`train/archive/`、`train/backup_model/`、`_package_tmp/`、Docker 镜像等大型运行产物。
- 不破坏 monitor URL：`server_req_base_url` 必须保持 `http://127.0.0.1:${MONITOR_TRPC_PORT}`。

## Verification Strategy
> ZERO HUMAN INTERVENTION - 所有验证由 agent 执行。

- 测试策略：tests-after + dry-run + py_compile + Docker/benchmark 运行验证。
- Benchmark 标准：
  - 快速门槛：`--episodes-per-map 10`，地图 `[4,7]`，共 20 局。
  - 确认门槛：若分数距离 900 在 ±20 内，或死亡率/方差偏高，则跑 `--episodes-per-map 20`。
  - 硬规则：若 20 局均分刚好超过 900 但 `p10 < 850`、任一地图均分 `<900`、battery/collision fail rate 偏高、或方差明显异常，不得宣告成功，必须自动升级样本量确认。
  - 最终门槛：若决定最终模型且时间允许，跑 `--episodes-per-map 50`。
- 闭环停止条件：只有 benchmark combined average clean score **>900** 才能停止并进入最终报告；否则必须继续下一轮优化或升级策略层级。
- 优化优先级：
  1. **纯正面收益**：不会改变策略行为或只修正选择/保存/分析错误，例如 best checkpoint 过滤、benchmark、日志、死亡回放、分析工具、checkpoint ranking。
  2. **低风险行为修复**：小范围 planner/target gating/safety buffer，不改模型、不改大规模 reward。
  3. **奖励新增或改写**：允许新增 reward component、重写局部 reward credit，但必须先用日志证明 failure/reward 链路，并做 A/B benchmark。
  4. **小幅重构**：为了可观测性、策略模块解耦、benchmark 控制、配置注入、重复逻辑收敛而做的小范围结构整理。
  5. **网络改动**：最后手段；必须先有证据说明现有特征/规则/奖励不足以表达问题，并保持可回退、可验证。
- 接受门槛：
  - combined holdout avg 提升 `>=20`，或达到并超过 `900`；
  - completed rate 绝对下降不超过 `0.05`；
  - clean_per_step 下降不超过 `2%`。
- 拒绝门槛：
  - battery fail rate 绝对增加 `>0.05`；
  - collision fail rate 绝对增加 `>0.02`；
  - `clean_score > 850` 的高分电池死亡增加；
  - benchmark 检测到 holdout 泄漏或 eval 修改模型文件。

## Execution Strategy

### 执行波次
- **Wave 0**：T0 参考 `linux-LTSPPO-charge-constraint`，确定本地 benchmark/日志适配契约。
- **Wave 1**：T1 baseline、T2 benchmark 基础设施、T7 changelog/log 规则。
- **Wave 2**：T3 baseline benchmark、T8 死亡回放深度分析。
- **Wave 3**：T4 completed-only best 修复、T5 coverage return buffer 修复；二者可并行实现但必须分开 commit/评估。
- **Wave 4**：T6 自动闭环执行与接受/回退。
- **Wave 5（按需升级）**：T10 基于证据的奖励/重构/网络升级优化；只要 T6 未达 `>900` 就持续进入，不以轮数停止。
- **Wave 6**：T9 最终 checkpoint 选择报告；只有 benchmark combined avg `>900` 后进入。

### 依赖矩阵
| Task | 依赖 | 阻塞 |
|------|------|------|
| T0 | none | T2, T3, T8, T6 |
| T1 | none | T3, T6, T9 |
| T2 | T0 | T3, T8, T6 |
| T3 | T1, T2 | T4/T5 的对比依据 |
| T4 | T1, T3 | T6 |
| T5 | T1, T3 | T6 |
| T6 | T2, T3, T4, T5, T8 | T10, T9 |
| T7 | none | 所有 changelog/log 验收 |
| T8 | T2, T3 | T6, T9 |
| T9 | T6, T8, T10(如触发升级) | Final Verification |
| T10 | T6, T8 | T9 |

## TODOs

- [x] 0. 前置：参考 `linux-LTSPPO-charge-constraint` 构建本地 Benchmark/日志方案

  **要做什么**：先用 `git ls-remote --heads origin "*linux-LTSPPO-charge-constraint*"` 探测远端分支，再获取并检查远端分支 `linux-LTSPPO-charge-constraint` 的 benchmark、charge constraint、日志、分析工具设计。当前已确认远端存在 `refs/heads/linux-LTSPPO-charge-constraint`；如果本地没有 `origin/linux-LTSPPO-charge-constraint` 对象，执行只读意图的 `git fetch origin linux-LTSPPO-charge-constraint:refs/remotes/origin/linux-LTSPPO-charge-constraint` 以便读取文件，不得 checkout/merge 到当前分支。重点提取 benchmark runner 的控制方式、固定配置注入方式、日志字段、分析 report schema、死亡/充电约束相关指标。生成 `train/context/LTSPPO_BENCHMARK_ADAPTATION_YYYYMMDD.md`，说明哪些设计照搬、哪些因 `win_YJY` 本地 Docker/配置差异需要适配。随后 T2 按该适配说明实现本地 benchmark。
  **固定本地 benchmark 要求**：`max_step=1000`，`battery_max=150`，`robot_count=4`，`charger_count=3`，地图 `[4,7]`，每图 `10` 局，共 `2*10`。核心输出不是只给平均分，而是给 AI 能详细分析的结构化日志。
  **禁止**：不得 checkout/merge `linux-LTSPPO-charge-constraint`；不得把该分支的算法/网络改动整体带入；不得改变当前训练地图；不得把远端分支的 benchmark 参数照搬成非固定配置。

  **推荐 Agent**：`deep` + `git-master`。  
  **并行**：NO | Wave 0 | Blocks: T2/T3/T8/T6  
  **参考**：远端 `refs/heads/linux-LTSPPO-charge-constraint`；当前计划固定 benchmark 配置；`train/context/HANDOFF_20260425_WIN_YJY.md:214-268`。

  **验收标准**：
  - [ ] 先用 `ls-remote` 确认远端分支存在；再能读取 `origin/linux-LTSPPO-charge-constraint` 或记录 fetch/读取失败的具体原因。
  - [ ] `LTSPPO_BENCHMARK_ADAPTATION_YYYYMMDD.md` 列出可复用的 benchmark/logging 设计与本地适配差异。
  - [ ] 文档明确本地 benchmark 固定为 1000 步、150 电量、4 机器人、3 充电桩、`2*10`。
  - [ ] 文档定义 AI 分析日志最小字段集：episode id、map id、score、done/fail reason、step、action、planner mode/target、battery、charger distance/slack、reward components、death replay path、checkpoint id。

  **QA 场景**：
  ```
  Scenario: LTSPPO 分支只读参考完成
    Tool: Bash
    Steps: 运行 `git ls-remote --heads origin "*linux-LTSPPO-charge-constraint*"` 和 `git branch -a --list "*linux-LTSPPO-charge-constraint*"`；若缺失远端对象，fetch 到 remote-tracking ref；列出该分支 benchmark/log 相关文件；生成适配文档。
    Expected: 当前分支仍是 `win_YJY`；没有 merge/checkout；适配文档存在。
    Evidence: .sisyphus/evidence/task-0-ltsppo-reference.txt

  Scenario: 固定 benchmark 参数未被远端默认覆盖
    Tool: Bash
    Steps: 检查适配文档中的本地 benchmark 参数。
    Expected: 明确写入 `max_step=1000`、`battery_max=150`、`robot_count=4`、`charger_count=3`、maps `[4,7]`、每图 10 局。
    Evidence: .sisyphus/evidence/task-0-fixed-benchmark-contract.md
  ```

  **Commit**：YES | `docs(benchmark): adapt LTSPPO charge constraint benchmark design`

- [x] 1. 固定当前 baseline 与 git 回退锚点

  **要做什么**：记录当前 branch、commit、dirty status、源码 diff、模型文件大小/hash、mtime、resume meta、固定训练配置与 checkpoint 路径。生成 `train/context/BASELINE_YYYYMMDD_win_YJY.md`。这是硬 gate：T1 未完成前，不得跑 T3 baseline benchmark，不得执行 T4/T5/T10 行为改动。不得 reset。若当前 dirty 源码就是本轮实验起点，则在排除大型产物后做 baseline commit。
  **禁止**：不得 `git reset`；不得删除 checkpoint；不得提交 `train/log/`、`train/archive/`、`train/backup_model/`、Docker 镜像。

  **推荐 Agent**：`quick` + `git-master`。  
  **并行**：YES | Wave 1 | Blocks: T3/T6/T9  
  **参考**：`HANDOFF_20260425_WIN_YJY.md:9-49`、`AGENTS.md:36-47`、`AGENTS.md:101-107`。

  **验收标准**：
  - [ ] baseline markdown 包含 branch、commit、dirty 文件、固定 config、checkpoint metadata、rollback anchor。
  - [ ] baseline markdown 包含模型文件 hash/mtime，后续 benchmark 以此检测 eval 是否污染模型。
  - [ ] 证据中保存 `git status --short --branch`。
  - [ ] 没有大型 runtime artifact 被 staged。

  **QA 场景**：
  ```
  Scenario: baseline 报告完整
    Tool: Bash
    Steps: 运行 `git status --short --branch`、`git log -1 --oneline`、`python train/resume_best.py latest`，检查 baseline markdown。
    Expected: 报告包含训练/留出地图、dirty 文件、checkpoint 元信息与回退说明。
    Evidence: .sisyphus/evidence/task-1-baseline.md

  Scenario: 未 stage 大型产物
    Tool: Bash
    Steps: 运行 `git status --short` 和 `git diff --cached --name-only`。
    Expected: 没有 `train/log/`、`train/archive/`、`train/backup_model/`、`_package_tmp/`、Docker image。
    Evidence: .sisyphus/evidence/task-1-no-bulky-artifacts.txt
  ```

  **Commit**：YES | `chore(baseline): record win_YJY experiment anchor`

- [x] 2. 新增固定 `[4,7]` Holdout Benchmark Runner 与报告契约

  **要做什么**：在 T0 的 `LTSPPO_BENCHMARK_ADAPTATION_YYYYMMDD.md` 基础上新增 `train/run_holdout_benchmark.py` 与 `train/analyze_holdout_benchmark.py`。runner 必须只跑 `[4,7]`，固定 `robot_count=4`、`charger_count=3`、`max_step=1000`、`battery_max=150`，默认每图 10 局。支持 `--dry-run`、`--maps 4,7`、`--episodes-per-map N`、`--checkpoint code/model.ckpt-resume.pkl`、`--output train/context/HOLDOUT_BENCHMARK_YYYYMMDD_HHMM.json`、`--detail-log-dir train/holdout_detail_logs/RUN_ID`。context 只保存小型 summary JSON/Markdown；详细 step/action/reward/death replay 日志可写入 `train/holdout_detail_logs/` 或 `.sisyphus/evidence/`，不得放入 `train/context/`。runner 必须是 inference-only：优先使用不触发保存逻辑的评估入口；若必须复用训练 workflow，则必须显式禁用 `latest/resume/best/session_best` 保存路径。记录 `code/latest_model.pkl`、`code/model.ckpt-resume.pkl`、`code/model.ckpt-resume.meta.json` 运行前后 hash/mtime，若 eval 修改模型文件则失败。
  **禁止**：不得训练 `[4,7]`；不得永久修改 `train_env_conf.toml`；不得整体 merge `origin/linux`。

  **推荐 Agent**：`unspecified-high`。  
  **并行**：YES | Wave 1 | Blocked By: T0 | Blocks: T3/T8/T6  
  **参考**：T0 输出 `train/context/LTSPPO_BENCHMARK_ADAPTATION_YYYYMMDD.md`、`HANDOFF_20260425_WIN_YJY.md:214-304`、`train/benchmark_report.py:14`、`code/agent_ppo/utils/archive_analysis.py:51/173`、`train/collect_data.py:9-36`、`train/tb_writer.py:32-67`。

  **验收标准**：
  - [ ] `python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 10 --dry-run` 退出 0。
  - [ ] analyzer 输出 combined、map 4、map 7 三组指标。
  - [ ] JSON schema 包含 `run_id/checkpoint/maps/episodes_per_map/fixed_config/combined/per_map`。
  - [ ] 详细日志 schema 至少包含 `episode_id/map_id/step/action/planner_mode/planner_target/battery/charger_distance/return_slack/reward_components/fail_reason/death_replay_path/checkpoint_id`。
  - [ ] dry-run 验证训练地图仍排除 4 和 7。
  - [ ] runner 会检查 eval 不修改模型文件，并明确证明没有触发训练保存路径。

  **QA 场景**：
  ```
  Scenario: holdout dry-run 无泄漏
    Tool: Bash
    Steps: 运行 `python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 10 --dry-run --output train/context/HOLDOUT_BENCHMARK_dryrun.json`。
    Expected: 输出只包含 benchmark maps `[4,7]`，训练 maps 仍是 `[1,2,3,5,6,8,9,10]`。
    Evidence: .sisyphus/evidence/task-2-holdout-dryrun.json

  Scenario: 详细日志可供 AI 分析
    Tool: Bash
    Steps: 运行 dry-run 或最小 1 局 benchmark，指定 `--detail-log-dir .sisyphus/evidence/task-2-detail-logs`，检查 episode summary 与 step/action trace schema。
    Expected: 日志包含 planner、reward、battery、charger、fail/death replay 关联字段；summary 留在 context，原始 detail 不写入 context。
    Evidence: .sisyphus/evidence/task-2-detail-log-schema.json

  Scenario: benchmark 不污染模型文件
    Tool: Bash
    Steps: 记录 `code/latest_model.pkl`、`code/model.ckpt-resume.pkl`、`code/model.ckpt-resume.meta.json` 的 hash/mtime；运行 dry-run 或 1 局 inference-only benchmark；再次记录 hash/mtime。
    Expected: 三个文件 hash/mtime 均未变化；runner 日志说明保存路径已禁用或未调用。
    Evidence: .sisyphus/evidence/task-2-inference-only-model-mutation.txt

  Scenario: 拒绝训练地图泄漏
    Tool: Bash
    Steps: 运行 `python train/run_holdout_benchmark.py --maps 1,4 --episodes-per-map 1 --dry-run`。
    Expected: 非 0 退出，并明确报错 holdout maps 不得包含训练地图 1。
    Evidence: .sisyphus/evidence/task-2-leakage-reject.txt
  ```

  **Commit**：YES | `feat(benchmark): add fixed holdout evaluation runner`

- [x] 3. 跑当前状态的不可变 baseline benchmark

  **要做什么**：在 T2 commit 后，使用当前 baseline checkpoint 跑 `[4,7]`，每图 10 局。生成 `HOLDOUT_BENCHMARK_BASELINE_*.json` 与 `HOLDOUT_ANALYSIS_BASELINE_*.md`。分析必须包含：combined avg、per-map avg、completed rate、battery fail rate、collision fail rate、clean_per_step、p10/p50/p90、高分电池死亡、模型文件是否被 eval 修改。若 Docker/runtime 失败，必须记录错误和下一步，不得假装通过。
  **禁止**：baseline 前不得先做 T4/T5 算法修复。

  **推荐 Agent**：`deep`。  
  **并行**：NO | Wave 2 | Blocks: T4/T5 对比  
  **参考**：`HANDOFF_20260425_WIN_YJY.md:322-356`、`code/agent_ppo/workflow/train_workflow.py:410/778`、`code/agent_ppo/utils/experiment_archive.py:171`、`code/agent_ppo/utils/archive_analysis.py:173`。

  **验收标准**：
  - [ ] JSON 至少包含 map 4 和 map 7 各 10 局，除非 runtime 失败且有报告。
  - [ ] Markdown 包含所有关键指标与死亡类型统计。
  - [ ] 若 combined avg `>900` 但任一地图均分 `<900`、`p10 <850`、死亡率偏高或波动异常，报告必须自动要求追加确认 benchmark，而不是进入最终完成。
  - [ ] `CHANGELOG.md` 有 baseline benchmark 记录。

  **QA 场景**：
  ```
  Scenario: baseline 完成 2x10
    Tool: Bash
    Steps: 跑 holdout benchmark 与 analyzer。
    Expected: JSON 共 20 局，per_map keys 正好是 4 和 7，combined avg 为数字，模型文件未被修改。
    Evidence: .sisyphus/evidence/task-3-baseline-benchmark.json

  Scenario: runtime 失败被记录
    Tool: Bash
    Steps: benchmark 非 0 时保存 stdout/stderr，并运行 `docker ps --format "table {{.Names}}\t{{.Status}}"`。
    Expected: 证据说明失败原因和下一步。
    Evidence: .sisyphus/evidence/task-3-runtime-failure.md
  ```

  **Commit**：YES | `test(benchmark): record fixed holdout baseline`

- [x] 4. 实现“只有 completed 局能晋升 best checkpoint”

  **要做什么**：修改 `code/agent_ppo/workflow/train_workflow.py`，确保只有 `fail_reason == "completed"` 时才能设置 `self.is_new_best = True` 或提高 `self.best_robust_score`。失败局仍然参与训练、archive、monitor、死亡回放和统计，但不能晋升 best，也不能因为滚动分数提高而阻塞后续 completed 局。
  **禁止**：不得丢弃失败局训练样本；不得移除死亡回放；本任务不调整 completed 局得分公式。

  **推荐 Agent**：`quick`。  
  **并行**：YES | Wave 3 | Blocks: T6  
  **参考**：`HANDOFF_20260425_WIN_YJY.md:156-174`、`train_workflow.py:489/778`。

  **验收标准**：
  - [ ] `python -m py_compile code/agent_ppo/workflow/train_workflow.py` 通过。
  - [ ] 源码检查确认失败局不能设置 `is_new_best=True`，不能提高 `best_robust_score`。
  - [ ] `archive.log_episode_summary(...)` 与 `_log_death_replay(...)` 仍保留。
  - [ ] `CHANGELOG.md` 更新。

  **QA 场景**：
  ```
  Scenario: 失败局不能晋升 best
    Tool: Bash
    Steps: 用静态检查脚本确认 best promotion 分支被 `fail_reason == "completed"` 保护。
    Expected: 没有失败局无条件晋升 best 的路径。
    Evidence: .sisyphus/evidence/task-4-best-guard.txt

  Scenario: 失败局 telemetry 保留
    Tool: Bash
    Steps: 搜索 `archive.log_episode_summary`、`battery_fail`、`collision_fail`、`_log_death_replay`。
    Expected: 所有 telemetry 调用仍存在。
    Evidence: .sisyphus/evidence/task-4-telemetry-preserved.txt
  ```

  **Commit**：YES | `fix(workflow): prevent failed episodes from promoting best checkpoints`

- [x] 5. 为 coverage target gating 增加返航缓冲

  **要做什么**：修改 `code/agent_ppo/algorithm/algorithm.py`，在 `BASE_RETURN_MARGIN` 附近新增 `COVERAGE_RETURN_BUFFER = 8.0`。仅在 `_select_coverage_target(...)` 的 coverage 候选目标电量预算判断中使用：`charger_need = self._heuristic_charger_distance(pos) + self.COVERAGE_RETURN_BUFFER`。不得改变 charge-mode A*、不得全局提高 `BASE_RETURN_MARGIN`。
  **禁止**：不改 `_heuristic_charger_distance(...)` 语义；不改网络/残差 alpha。

  **推荐 Agent**：`quick`。  
  **并行**：YES | Wave 3 | Blocks: T6  
  **参考**：`HANDOFF_20260425_WIN_YJY.md:175-205`、`algorithm.py:417/656/1569`。

  **验收标准**：
  - [ ] `python -m py_compile code/agent_ppo/algorithm/algorithm.py` 通过。
  - [ ] `COVERAGE_RETURN_BUFFER = 8.0` 存在。
  - [ ] buffer 只用于 coverage target gating。
  - [ ] `BASE_RETURN_MARGIN` 保持 `28.0`，除非另有证据。
  - [ ] `CHANGELOG.md` 更新。

  **QA 场景**：
  ```
  Scenario: buffer 作用域正确
    Tool: Bash
    Steps: 搜索 `COVERAGE_RETURN_BUFFER`、`_select_coverage_target`、`_heuristic_charger_distance`。
    Expected: buffer 仅用于 coverage target gating，不进入 charge-mode A* 或全局阈值。
    Evidence: .sisyphus/evidence/task-5-buffer-scope.txt

  Scenario: planner 编译通过
    Tool: Bash
    Steps: 运行 `python -m py_compile code/agent_ppo/algorithm/algorithm.py`。
    Expected: Exit 0。
    Evidence: .sisyphus/evidence/task-5-pycompile.txt
  ```

  **Commit**：YES | `fix(planner): add return buffer for coverage target gating`

- [x] 6. 执行 Train → Benchmark → Analyze → Accept/Rollback 持续闭环

  **要做什么**：新增 `train/run_closed_loop_iteration.py` 或 `train/context/RUNBOOK_closed_loop_YYYYMMDD.md`。流程：记录 git anchor → 训练 20 分钟 quick 或 50 分钟 full → 跑 `[4,7]` benchmark → 跑 analyzer → 按门槛决定接受/拒绝 → 接受则保留 commit，拒绝则 `git revert <commit>` 并记录。闭环不能因为达到固定轮数就停止；必须持续迭代，直到 `[4,7]` benchmark combined avg **>900**。如果连续 3 轮未达到 >900，不停止，而是升级优化层级：从纯正面收益/低风险修复升级到奖励新增或改写，再到小幅重构，最后才考虑网络改动。优先分别评估 T4、T5；如果已连续提交，则按 commit hash 分别 benchmark。
  **禁止**：不得一次应用多个新行为改动；benchmark 报 holdout 泄漏或 eval 修改模型文件时必须暂停该轮、修复基础设施后继续闭环；不得把“连续无提升”当作完成理由。

  **推荐 Agent**：`deep` + `git-master`。  
  **并行**：NO | Wave 4 | Blocks: T9  
  **参考**：`HANDOFF_20260425_WIN_YJY.md:305-410`、`train/AGENTS.md:18-38`、`AGENTS.md:70-91`。

  **验收标准**：
  - [ ] 闭环停止条件明确为 benchmark combined avg `>900`；没有固定最大轮数停止条件。
  - [ ] 每轮报告包含 commit hash、改动摘要、训练时长、benchmark 路径、接受/拒绝、回退动作。
  - [ ] 拒绝的改动通过 git revert 回退，并记录 revert hash。
  - [ ] 连续 3 轮未超过 900 时，报告自动升级到下一优化层级，而不是停止。
  - [ ] 达到 `>900` 且结果接近/高波动时，会跑确认 benchmark。

  **QA 场景**：
  ```
  Scenario: 闭环 dry-run 不启动 Docker
    Tool: Bash
    Steps: 运行 `python train/run_closed_loop_iteration.py --dry-run --until-score-gt 900` 或验证 runbook 命令。
    Expected: 输出 git anchor、训练命令、benchmark 命令、analyzer 命令、rollback 命令、优化升级阶梯，不启动容器；不出现固定 3 轮后停止的逻辑。
    Evidence: .sisyphus/evidence/task-6-loop-dryrun.txt

  Scenario: 拒绝改动有明确回退命令
    Tool: Bash
    Steps: 用一份 battery_fail_rate 超标的模拟 metrics JSON 跑 decision mode。
    Expected: 输出 `REJECT` 与具体 `git revert <commit>`。
    Evidence: .sisyphus/evidence/task-6-reject-rollback.txt
  ```

  **Commit**：YES | `feat(loop): add closed-loop benchmark optimization runner`

- [x] 7. 维护 Changelog 与详细实验日志

  **要做什么**：每次实现、benchmark、bug 排查、接受/拒绝优化后追加 `train/context/CHANGELOG.md`。多文件闭环工作创建 `train/context/LOG_YYYYMMDD_closed_loop_holdout.md`，总结基础设施、命令、schema、迭代、决策和回退点。
  **禁止**：不得把原始 Docker logs、checkpoints、zip、巨大 metric dumps 放进 `train/context/`。

  **推荐 Agent**：`writing`。  
  **并行**：YES | Wave 1+  
  **参考**：`train/context/AGENTS.md:28-39`、`AGENTS.md:93-99`。

  **验收标准**：
  - [ ] 每个代码/infra commit 都有 changelog 一句话。
  - [ ] 多文件闭环有详细 LOG 文档。
  - [ ] context 只含摘要与路径，不含大型原始输出。

  **QA 场景**：
  ```
  Scenario: changelog 覆盖所有迭代
    Tool: Bash
    Steps: 对比 `git log --oneline` 中闭环相关 commits 与 `CHANGELOG.md`。
    Expected: 每个相关 commit 都有日期摘要。
    Evidence: .sisyphus/evidence/task-7-changelog-coverage.txt

  Scenario: context 保持轻量
    Tool: Bash
    Steps: 列出 `train/context` 中大文件和扩展名。
    Expected: 没有 raw log/checkpoint/zip/大型 dump。
    Evidence: .sisyphus/evidence/task-7-context-lightweight.txt
  ```

  **Commit**：YES | `docs(context): document closed-loop holdout process`

- [x] 8. 增加 Holdout 与死亡回放深度分析

  **要做什么**：增强 analyzer，结合 benchmark JSON、archive episode summary、`death_replay.*.jsonl`，并允许新增需要的分析工具。分类：首个充电桩未发现、已知充电桩但返航触发太晚、路径预算过于乐观、NPC/path blocked、重复 invalid move、高分电池死亡、后期电池死亡、unknown。除死亡分类外，还要分析 per-map 分数分布、reward component 趋势、clean_per_step、charger arrival step、battery slack、动作分布、invalid move、checkpoint ranking、训练 20/50 分钟曲线。输出 Markdown 与 JSON，并给出唯一下一步建议及其优化层级：纯正面收益、低风险、奖励新增/改写、小幅重构、网络改动。
  **禁止**：analyzer 不得在无证据时自动改 reward/planner/network；它只能提出有证据的一个下一步建议。如果证据不足但分数未 >900，则输出 `NEED_MORE_DATA` 和需要采集的具体日志/指标，而不是停止闭环。

  **推荐 Agent**：`unspecified-high`。  
  **并行**：YES | Wave 2 | Blocks: T6/T9  
  **参考**：`HANDOFF_20260425_WIN_YJY.md:332-356`、`code/agent_ppo/workflow/train_workflow.py:410`、`code/agent_ppo/utils/archive_analysis.py:77-123`、`train/tb_writer.py:46-55`。

  **验收标准**：
  - [ ] 输出 failure classification counts 与例子。
  - [ ] 能识别高分/后期电池死亡。
  - [ ] 输出唯一下一步建议、所属优化层级、证据路径；若证据不足且未达 >900，输出 `NEED_MORE_DATA`。
  - [ ] 缺失 death replay 时能降级处理。
  - [ ] 分析器支持多种日志源：benchmark JSON、archive JSONL、AISRV GAMEOVER、training_metrics、death_replay、TensorBoard 解析输出或自建中间 JSON。

  **QA 场景**：
  ```
  Scenario: replay-backed 电池死亡可分类
    Tool: Bash
    Steps: 用含 death replay 的 benchmark/archive 跑 analyzer。
    Expected: Markdown 包含分类计数和至少一个轨迹例子。
    Evidence: .sisyphus/evidence/task-8-replay-analysis.md

  Scenario: death replay 缺失时不崩溃
    Tool: Bash
    Steps: 用只有 benchmark JSON、无 replay 路径的数据跑 analyzer。
    Expected: 报告说明 replay 缺失，并回退到 fail_reason/map/score/reward/metrics 汇总；如仍不足则输出 `NEED_MORE_DATA`。
    Evidence: .sisyphus/evidence/task-8-missing-replay.txt
  ```

  **Commit**：YES | `feat(analysis): add holdout death replay diagnostics`

- [x] 10. 按证据升级优化：奖励新增/改写、小幅重构、必要时网络改动

  **要做什么**：当 T4/T5 等纯正面收益与低风险修复不能让 `[4,7]` benchmark combined avg **>900** 时，继续闭环，不停止。根据 T8 analyzer 的证据按优先级生成并执行下一类改动：
  1. **奖励新增/改写**：不仅限于调权重；可新增 battery budget reward、late-return penalty、charger departure discipline、route slack potential、high-score death suppression、per-map weak-signal reward 等，但必须有日志/死亡回放证据证明问题链路。
  2. **小幅重构**：可重构 planner/reward/analysis 配置注入、目标选择、死亡回放字段、benchmark 控制、feature/reward component 输出，目的是提升可验证性或降低 bug 风险。
  3. **网络改动**：最后手段。允许小范围网络结构改动，例如轻量层宽调整、增加输入处理头、轻量记忆/统计特征接口，但必须单独计划、单独 commit、单独 benchmark，并验证 checkpoint 兼容/不兼容处理。
  每个升级改动都必须包含：假设、证据、具体 diff 范围、预期指标变化、回退点、benchmark 结果。
  **禁止**：不得一次做 reward + planner + network 混合大改；不得无 benchmark 正收益就保留；不得只凭直觉改 reward；不得因为网络改动麻烦就假装完成。

  **推荐 Agent**：`deep`；网络改动时使用 `oracle` 先审查。  
  **并行**：NO | Wave 5 | Blocks: T9  
  **参考**：T8 analyzer 输出、`code/agent_ppo/feature/preprocessor.py` reward_process、`code/agent_ppo/algorithm/algorithm.py` planner、`code/agent_ppo/model/model.py` 网络结构、`code/agent_ppo/conf/conf.py` 维度/超参。

  **验收标准**：
  - [ ] 每个升级改动都有证据来源和假设，不允许“凭感觉”。
  - [ ] 奖励新增/改写不只调权重；如果只调权重，必须说明为何足够。
  - [ ] 小幅重构有明确验证收益：减少歧义、增加可观测性、修复配置/日志问题或降低 bug 风险。
  - [ ] 网络改动前必须记录 checkpoint 兼容性策略，并通过 py_compile 与 benchmark。
  - [ ] 未达到 `>900` 时继续提出下一轮，不以轮数耗尽为停止理由。

  **QA 场景**：
  ```
  Scenario: reward 改写必须有证据和正收益
    Tool: Bash
    Steps: 检查 iteration report 中 reward 改动的 hypothesis、evidence、before/after benchmark。
    Expected: 报告显示改动前的 failure/reward 证据，改动后的 benchmark 不触发拒绝门槛。
    Evidence: .sisyphus/evidence/task-10-reward-positive.md

  Scenario: 网络改动不能绕过优先级
    Tool: Bash
    Steps: 若出现 model/model.py 或 feature dimension 改动，检查是否存在 Oracle 审查、checkpoint 兼容说明、独立 benchmark 和回退 commit。
    Expected: 网络改动仅在低优先级策略不足后出现，且有完整验证。
    Evidence: .sisyphus/evidence/task-10-network-guard.md
  ```

  **Commit**：YES | Message 按改动类型：`feat(reward): ...` / `refactor(planner): ...` / `feat(model): ...` | Files: 由 analyzer 指定，必须包含 `train/context/CHANGELOG.md` 与 iteration report。

- [x] 9. 最终 checkpoint 选择与提交准备报告

  **要做什么**：只有当 `[4,7]` benchmark combined avg **>900** 时，才进入最终 checkpoint 选择。根据 robust ranking 选择 checkpoint，生成 `train/context/FINAL_SELECTION_YYYYMMDD.md`，包含 checkpoint path/id、quick 与 confirmation benchmark、map 4/7 指标、训练地图 sanity、死亡统计、git commit hash、rollback point、打包命令与 monitor URL 验证。若未达标，不生成“最终成功”报告，只生成 iteration report 并继续 T6/T10 闭环。
  **禁止**：不得在本计划中自动 package/sign/submit；不得选择失败局晋升的 checkpoint。

  **推荐 Agent**：`writing` + `git-master`。  
  **并行**：NO | Wave 6  
  **参考**：`HANDOFF_20260425_WIN_YJY.md:396-410`、`train/package_model.py`、`train/AGENTS.md:47-58`、`code/agent_ppo/utils/archive_analysis.py:51`。

  **验收标准**：
  - [ ] 报告明确 `>900` 已通过；未通过时不得作为最终选择报告。
  - [ ] 接近/通过门槛时包含 confirmation benchmark 或明确计划。
  - [ ] 报告有 checkpoint、打包命令、失败率、高分电池死亡、git hash、rollback。

  **QA 场景**：
  ```
  Scenario: 最终报告可直接决策
    Tool: Bash
    Steps: 检查 `FINAL_SELECTION_YYYYMMDD.md` 并运行其中引用的 analyzer 命令。
    Expected: 包含 checkpoint、commit hash、map 4/7 指标、门槛状态、打包命令、回退点。
    Evidence: .sisyphus/evidence/task-9-final-report.md

  Scenario: 未达标时不伪造成功
    Tool: Bash
    Steps: 用低于或等于 900 的 metrics 尝试生成最终报告。
    Expected: 工具拒绝生成最终成功报告，改为生成 iteration report，并列出下一步证据支持改动或 `NEED_MORE_DATA`。
    Evidence: .sisyphus/evidence/task-9-not-met.md
  ```

  **Commit**：YES | `docs(selection): record holdout benchmark checkpoint decision`

## Final Verification Wave
> 所有实现任务完成后，4 个 review agent 并行运行；全部通过后把结果呈现给用户，等待用户明确 “okay”。不得自动完成。

- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high（Docker/benchmark 执行；不需要 Playwright，除非测试 dashboard UI）
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy

建议 commit 顺序：
1. `docs(benchmark): adapt LTSPPO charge constraint benchmark design`
2. `chore(baseline): record win_YJY experiment anchor`
3. `feat(benchmark): add fixed holdout evaluation runner`
4. `test(benchmark): record fixed holdout baseline`
5. `fix(workflow): prevent failed episodes from promoting best checkpoints`
6. `fix(planner): add return buffer for coverage target gating`
7. `feat(analysis): add holdout death replay diagnostics`
8. `feat(loop): add closed-loop benchmark optimization runner`
9. `docs(context): document closed-loop holdout process`
10. 按需升级改动：`feat(reward): ...` / `refactor(planner): ...` / `feat(model): ...`
11. `docs(selection): record holdout benchmark checkpoint decision`

每次 commit 前必须：
- `git status --short`
- 检查 diff
- 排除大型 artifact
- 记录 benchmark/analysis 证据路径

拒绝改动时：
- 使用 `git revert <commit>`，不要手动半回退。
- 在 iteration report 和 changelog 中记录 revert hash。

## Success Criteria

- 立即目标：地图 `[4,7]` 各 10 局，combined 平均分 `>900`；达到之前不得停止闭环。
- 安全目标：battery fail rate 不恶化；collision fail rate 稳定；不出现重复高分电池死亡。
- 稳健目标：接近门槛或方差大时跑 20 局/图确认。
- 最终目标：向 handoff 的 `>=950` 推进，同时保持 completed rate 和失败率健康。
- 流程目标：每个改动 commit 或 revert；所有报告/changelog 存在；没有 holdout 泄漏；没有未经证据、审查和独立 benchmark 的网络结构改动。
