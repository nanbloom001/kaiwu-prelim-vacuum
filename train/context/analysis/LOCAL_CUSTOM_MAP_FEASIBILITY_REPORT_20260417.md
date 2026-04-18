# 本地自定义泛化地图可行性分析报告

日期：2026-04-17

结论先行：

- 本地将“自定义泛化地图”纳入训练集，技术上**可行**。
- 可行性的关键原因不是猜测，而是已经在运行中的 `gamecore` 容器里确认到：
  - 官方公开地图是**明文 JSON 资源**
  - 地图加载链路是**明文 Python + adapter**
  - 目前真正卡住自定义地图的主要是**本地校验白名单**，不是黑盒二进制
- 这条路适合作为：
  - 本地泛化训练增强
  - 路线规划 / recoverability / diagonal 优势验证
- 这条路**不应**被理解为线上比赛环境注入手段。它是本地训练研究手段。

---

## 1. 本次分析目标

目标不是继续讨论“理论上能不能”，而是确认三件事：

1. 官方地图到底是不是黑盒资源
2. 地图是如何按 `map_id` 加载的
3. 如果要在本地加入自定义地图，最小需要改哪些东西

本次结论来自对**正在运行的训练容器**和 **gamecore 容器** 的实际检查。

---

## 2. 已确认的事实

### 2.1 训练侧并不持有地图资源

在 `aisrv` 容器里，训练仓库只暴露到：

- `agent_ppo/conf/train_env_conf.toml`
- `map = [1..10]`
- `map_random = true/false`
- `robot_count / charger_count / max_step / battery_max`

也就是说，训练侧配置的是：

- “本局使用哪些地图编号”

而不是：

- “地图文件路径”
- “地图拓扑内容”

这说明地图内容不在 agent 仓库中，而在环境侧。

### 2.2 当前训练实际使用的是 `kaiwu_env_proxy`，不是本地 `DirectEnv`

在 `aisrv` 容器中确认到：

- `/root/tools/conf/project_default.toml` 中：
  - `aisrv_framework = "kaiwu_env_proxy"`

因此当前训练模式下：

- `aisrv` 不直接在训练容器里实例化业务环境
- 而是通过 `kaiwu_env_proxy` 远程连接 `gamecore` 容器

这意味着：

- 真正的地图资源
- 地图 loader
- runtime

都应在 `gamecore` 容器的 `kaiwu_arena / kaiwu_env` 侧。

### 2.3 官方公开地图已经以明文 JSON 形式存在于 `gamecore`

在 `gamecore` 容器内确认到以下目录：

- `/data/projects/kaiwu_arena/robot_vacuum/simulator/map_json/`

其中存在：

- `robot_vacuum_map_1.json`
- `robot_vacuum_map_2.json`
- ...
- `robot_vacuum_map_10.json`

这说明：

- 公开地图不是黑盒二进制
- 不是只存在于 `.so` 内部
- 是可读、可复制、可生成的 JSON 资源

### 2.4 地图 JSON 的基本 schema 是明确的

抽查 `robot_vacuum_map_1.json` 后确认其核心字段包括：

- `name`
- `width`
- `height`
- `cells`
- `objects`
- `metadata`

其中：

- `width = 128`
- `height = 128`
- `cells` 长度为 `16384`

这与官方文档中 `128x128` 栅格地图设定一致。

### 2.5 地图加载逻辑是明文 Python，不是完全封死在 runtime 内

在 `gamecore` 容器内，核心加载逻辑位于：

- `/data/projects/kaiwu_arena/robot_vacuum/adapter/custom_adapter.py`

已确认的关键逻辑：

1. 根据 `map_id` 构造候选文件名
2. 支持这些命名格式：
   - `room_grid_map_{map_id}.json`
   - `robot_vacuum_map_{map_id}.json`
   - `map{map_id}.json`
3. 找到文件后执行：
   - `json.load(...)`
   - `self.runtime.load_map_data(json.dumps(map_data))`

这说明：

- runtime 接受外部 JSON 地图数据
- 不是只能从固定编译资源里读取
- 本地注入自定义地图并不需要重写整个 simulator

### 2.6 当前真正限制自定义地图的是环境校验

主要限制位于：

- `/data/projects/kaiwu_arena/environments/robot_vacuum/components/env_config_validator.py`
- `/data/projects/kaiwu_arena/environments/robot_vacuum/conf/game.yaml`

其中：

- `MAP_REPOSITORY = list(range(1, 16))`
- `available_maps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`

校验流程是：

1. `map_id` 必须在 `MAP_REPOSITORY`
2. `map_id` 还必须在 `available_maps`

因此本地自定义地图的主要障碍不是 loader，而是：

- `map_id` 白名单

### 2.7 地图选择链路是开放的

地图选取逻辑位于：

- `/data/projects/kaiwu_arena/environments/robot_vacuum/data_handler.py`

已确认：

- 环境会从 `env_conf["map"]` 中选择一个 `map_id`
- `map_random=true` 时随机选
- `map_random=false` 时按顺序轮换
- 最终把 `map_id` 写回 `env_conf`

然后：

- `/data/projects/kaiwu_arena/environments/robot_vacuum/env/standard_env.py`

会把这个 `map_id` 透传给底层环境 reset。

这意味着：

- 训练链路本身对 `map_id` 没有额外封死
- 只要 `map_id` 能通过校验，并且有对应 JSON 文件，就能走到 adapter loader

---

## 3. 结论：本地自定义地图训练集是否可行

结论是：

**可行，而且可行性较高。**

原因：

1. 地图资源已明文暴露
2. 地图 loader 已明文暴露
3. 训练链路已经支持按 `map_id` 调度
4. 当前真正需要放开的只是本地 validator 和 map whitelist

所以这不是：

- “需要破解黑盒引擎”
- “需要逆向专有二进制”

而更像是：

- “本地环境资源与校验策略扩展”

---

## 4. 最小可行改法

如果只追求“先证明本地自定义地图能进入训练”，最小改法如下。

### 4.1 新增自定义地图文件

在 `gamecore` 环境地图目录增加新文件：

- `/data/projects/kaiwu_arena/robot_vacuum/simulator/map_json/robot_vacuum_map_101.json`
- `/data/projects/kaiwu_arena/robot_vacuum/simulator/map_json/robot_vacuum_map_102.json`

命名建议使用 `101+`，不要覆盖 `1..10`。

原因：

- 避免污染官方公开图对照实验
- 避免 benchmark / 历史日志失真
- 与官方隐藏图编号语义解耦

### 4.2 放开本地环境对 map_id 的校验

至少需要改两处：

1. `env_config_validator.py`
   - 扩展 `MAP_REPOSITORY`
   - 允许 `101+`

2. `game.yaml`
   - 把 `available_maps` 扩展为包含自定义图编号

例如：

- `available_maps: [1,2,3,4,5,6,7,8,9,10,101,102]`

### 4.3 在训练配置中引入自定义地图

训练侧只需继续使用原有 `map` 列表机制，例如：

```toml
map = [1,2,3,4,5,6,7,8,9,10,101,102]
map_random = true
```

这样训练时会自然混采：

- 官方公开地图
- 本地自定义泛化地图

---

## 5. 为什么不建议直接覆盖官方图

虽然最省事的做法是：

- 直接把 `robot_vacuum_map_1.json` 之类替换掉

但我不建议这样做。

原因：

1. benchmark 与历史结果会失真
2. 你后面无法确认：
   - 是策略变了
   - 还是地图变了
3. 官方公开图对照价值会丢失

所以更稳的是：

- 保留 `1..10`
- 另起 `101+` 作为本地泛化图池

---

## 6. 主要风险与约束

### 6.1 JSON schema 不能乱

当前 runtime 接受的是特定 schema 的地图 JSON。

至少需要保证：

- `width`
- `height`
- `cells`
- `objects`
- `metadata`

结构完整且语义正确。

尤其是：

- `cells` 长度必须与 `128x128` 一致

### 6.2 对象布局必须合法

地图不仅仅是墙和空地，还涉及：

- charger
- spawn point
- dirt
- obstacle
- robot / NPC 可生成区域

如果只改障碍不改对象，可能会出现：

- 地图能加载
- 但 reset 逻辑或玩法初始化不合法

### 6.3 这条路线仅适用于本地训练研究

本地注入自定义地图：

- 不会影响官方隐藏评测环境
- 不会被线上评测继承

因此它的价值是：

- 提高本地训练的泛化压力
- 验证长期规划能力
- 暴露 return / contract / recoverability 的真实短板

而不是：

- 直接修改比赛环境

---

## 7. 这条路线为什么值得做

当前 LTSPPO 的主要挑战仍然集中在：

- battery fail
- late contract / late return
- long-horizon recoverability
- diagonal path efficiency

而官方公开 10 图虽然足够训练，但对泛化压力仍有限。

如果增加一批本地泛化图，可以更有针对性地构造：

- 长走廊地图
- 多房间地图
- 回充通道狭窄地图
- diagonal 优势特别明显的地图
- “高收益区离 charger 很远”的地图

这比单纯在现有 10 图里继续随机训练，更适合验证：

- `route_anchor`
- `contract`
- `return_action_teacher`
- `future_recoverability_score`

到底有没有学到真正的结构化能力。

---

## 8. 推荐实施顺序

### 阶段 1：最小接入验证

目标：

- 不先大规模造图
- 先验证自定义 `map_id` 和 JSON 是否真能跑通 reset / step

建议：

1. 复制一张官方地图为 `robot_vacuum_map_101.json`
2. 修改本地 validator 和 `available_maps`
3. 单独用 `map=[101]` 做一次本地 smoke test

验收标准：

- 环境 reset 成功
- obs 正常返回
- `extra_info["map_id"] == 101`
- 可正常 step

### 阶段 2：构造 2~4 张“有目的的泛化图”

建议优先做：

1. diagonal return 优势图
2. 长路径回充压力图
3. 多房间、回收半径受限图
4. charger 选择难题图

### 阶段 3：混入训练分布

训练配置改为：

- 公开图 + 自定义泛化图混训

而不是一开始就完全替换公开图。

---

## 9. 最终判断

最终判断如下：

- **本地自定义泛化地图训练集：可行**
- **核心入口已经找到：地图 JSON + adapter loader + validator**
- **最小代价不是重写 simulator，而是扩展本地地图编号和新增 JSON**

因此，如果目标是：

- 提升本地训练的泛化能力
- 更强地验证长期规划策略
- 为 `route_anchor / contract / recoverability` 提供更有压力的训练分布

那么这条路线值得继续推进。

---

## 10. 建议的下一步

建议直接进入实现前的最小验证阶段：

1. 新增一个最小自定义地图 `101`
2. 放开本地 map 校验
3. 让本地环境成功 reset / step 该地图

只要这一步跑通，后续把自定义地图纳入训练集就是工程推进问题，而不是可行性问题。

