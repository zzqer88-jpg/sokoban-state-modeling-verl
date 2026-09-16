# Sokoban VLM 状态建模 RL 数据集构造计划

## 1. 目标与边界

本数据集用于单独强化 Qwen3.5-4B 的视觉状态建模能力。模型不负责选动作，也不以通关为训练目标。

每个任务由环境直接生成一个多样化、合法的当前 Sokoban 状态，并提供 1-3 个 primitive actions。模型观察当前 RGB 图像，预测当前棋盘以及整段动作执行完毕后的最终状态：

```text
(当前 RGB, 环境给定动作块 a_t^1...a_t^k), k in [1, 3]
    -> 当前状态 s_t
    -> 动作块执行后的最终状态 s_{t+k}
```

其中动作由环境给定，VLM 不输出或修改动作。当前状态本身由状态生成器直接产生，不依靠从某个 reset 状态执行前缀动作来复现。

训练采用可验证奖励（RLVR）。数据集保存任务输入与隐藏真值，不要求提供自然语言标准答案。

## 2. 单条任务的数据格式

第一版固定使用 JSONL manifest 保存状态任务，不预存 PNG。每行表示一个确定性任务；训练时从 `current_state` 使用固定版本渲染器现场生成 RGB：

```json
{
  "schema_version": "sokoban-state-modeling-v2",
  "task_id": "train_00000001",
  "task_index": 0,
  "split": "train",
  "generator_version": "ragen-state-manifest-v9",
  "generation_seed": 18427,
  "accepted_attempt": 3,
  "sampler_profile": "main-v1",
  "resolved_generator_config": {
    "dim_room": [6, 6],
    "num_boxes": 1,
    "topology_steps": 20,
    "p_change_directions": 0.35,
    "reverse_search_depth": 300,
    "reverse_progress_bucket": "middle",
    "sampled_reverse_depth": 14,
    "player_placement_mode": "distance_bucket",
    "player_placement_bucket": "near"
  },
  "map_hash": "...",
  "current_state_hash": "...",
  "next_state_hash": "...",
  "query_actions": ["Right", "Right", "Down"],
  "task_hash": "...",
  "current_state": {
    "fixed": [
      [0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 0],
      [0, 1, 1, 2, 1, 0],
      [0, 1, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0]
    ],
    "entity": [
      [0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 2, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0]
    ],
    "grid": ["######", "#_P__#", "#____#", "#_XO_#", "#____#", "######"]
  },
  "next_state": {
    "fixed": [
      [0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 0],
      [0, 1, 1, 2, 1, 0],
      [0, 1, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0]
    ],
    "entity": [
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 2, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0]
    ],
    "grid": ["######", "#____#", "#___P#", "#_XO_#", "#____#", "######"]
  },
  "event_sequence": ["player_move", "player_move", "player_move"],
  "event_subtype_sequence": [null, null, null],
  "executed_action_count": 3,
  "terminated": false,
  "metadata": {
    "primary_state_category": "ordinary_solvable",
    "solver_status": "solvable",
    "shortest_solution_steps": 8,
    "wall_ratio": 0.44,
    "player_box_distance": 2,
    "player_box_distance_bucket": "near",
    "box_target_distance": 3,
    "query_length": 3,
    "state_attributes": {
      "is_player_adjacent_box": false,
      "is_near_completion": false,
      "has_box_on_target": false,
      "is_player_on_target": false,
      "is_narrow_or_high_wall": false
    }
  }
}
```

这里的 `resolved_generator_config` 是**这一条任务实际抽到的结果**，不是整个数据集的全局固定配置。它必须是具体值，才能准确复现状态与动作块转移真值。`accepted_attempt` 记录该任务最终接受的候选序号。真正决定多样性的分布保存在独立的 dataset-level `sampler_profile` 中。每条任务都从该 profile 重新采样，并把解析后的结果固化到 manifest。

训练 prompt 只能读取：

- 由隐藏 `current_state` 通过固定 `renderer_version` 现场生成的 RGB；
- `query_actions`；
- 固定的任务说明和输出格式。

`current_state`、`next_state`、`event_sequence`、`event_subtype_sequence`、`terminated` 和用于评分的 metadata 必须留在环境/奖励侧，不能进入模型上下文。

## 3. 状态表示

### 3.1 隐藏真值

数据集内部保存两层状态，保持与 Sokoban 引擎语义一致：

```text
fixed[r,c]  in {wall, floor, target}
entity[r,c] in {empty, player, box}
```

第一版整数编码固定为：

```text
fixed:  0=wall, 1=floor, 2=target
entity: 0=empty, 1=player, 2=box
```

对默认 `dim_room=[6,6]`，两个数组的 shape 都必须严格为 `[6,6]`。读取 manifest 时应校验每一行长度以及行数，禁止用省略的一行代替完整数组。

这样可以无损表示：

```text
box on target:    fixed=target, entity=box
player on target: fixed=target, entity=player
```

manifest 必须同时保存原始整数数组和规范化文本 `grid`，便于检查和训练输出构造；加载时二者不一致直接报错。

从现有引擎状态转成两层状态时，以运行时实际编码为准：`room_state` 中 `0=wall`、`1=floor`、`2=target`、`3=box on target`、`4=box on floor`、`5=player on floor`、`6=player on target`。`state/codec.py` 必须显式覆盖 0-6 的全部映射及往返测试，不能照抄生成器中可能已经过时的注释或假定另一套编号。

### 3.2 VLM 输出

VLM 使用单层复合符号网格，以缩短生成长度：

```text
# wall
_ floor
O target
X box
P player
* box on target
+ player on target
```

这里使用纯 ASCII 的 `*` 和 `+` 表示叠加状态，避免使用 `√` 等不常见字符增加 tokenizer 和解析器负担。`O` 是大写字母 O，不是数字 0。

规范输出：

```xml
<perception>
######
#_P__#
#____#
#_XO_#
#____#
######
</perception>
<prediction>
######
#__P_#
#____#
#_XO_#
#____#
######
</prediction>
```

标签语义与 VAGEN 保持一致：`<perception>` 是从当前 RGB 提取的状态，对应 `state_estimation`；`<prediction>` 是给定动作块执行后的最终状态，对应 `transition_prediction`。本任务不要求模型选动作，因此不需要 `<answer>`；也不强制生成显式 `<reasoning>`。解析器把复合符号还原成 `fixed + entity` 后再评分。坐标、相对关系和合法动作均由程序从棋盘派生，不要求模型重复输出，避免多个表示互相矛盾。

## 4. 直接生成多样化当前状态

### 4.1 生成原则

每条任务直接生成当前状态 `s0`。生成器输出完整的：

```text
room_fixed
room_state
player_position
box_mapping
```

不保存也不执行 `prefix_actions`。允许生成器内部使用反向 Sokoban 搜索来保证可解性，但应直接采样反向搜索轨迹中的状态作为最终生成结果，而不是先固定生成 reset 状态、再靠前缀动作到达目标状态。

### 4.2 需要开放的生成器参数

复制到实验目录后的 RAGEN 生成器必须扩展或暴露以下采样参数：

| 参数 | 作用 |
|---|---|
| `dim_room` | 地图尺寸 `[H, W]`；第一版只采样 6x6、7x7、8x8 正方形棋盘，不生成非正方形棋盘 |
| `num_boxes` | 箱子数量 |
| `topology_steps` | 控制房间形状复杂度 |
| `p_change_directions` | 控制走廊转向和拓扑形态 |
| `reverse_search_depth` | 反向搜索上限 |
| `sampled_reverse_depth` | 从反向轨迹的哪个深度采样当前状态 |
| `player_move_probability` | 反向生成后是否重新放置玩家 |
| `player_continue_probability` | 连续重新放置的概率 |
| `player_reposition_steps` | 玩家重新放置的最大步数 |
| `wall_ratio_range` | 墙体密度过滤范围 |
| `player_box_distance_range` | 玩家与最近箱子的距离带 |

这些参数不是整个数据集的一组固定常量。配置分成两层：

```text
dataset sampler profile：定义每个参数如何随机采样
task resolved_generator_config：记录这一条任务最终采到的具体参数
```

`player_move_probability`、`player_continue_probability` 和 `player_reposition_steps` 只为复现旧 RAGEN 行为保留。`main-v1` 固定把三者设为 `0, 0, 0`，不执行旧的随机玩家移动；玩家位置完全由后文的 shortest-walk distance bucket 直接选择。这样既能得到贴箱子和远距离样本，又不会让随机移动破坏已抽取的距离配额。

第一版 sampler profile 固定为：

```yaml
dataset_seed: 18427
generator_sampling:
  dim_room:
    values: [[6, 6], [7, 7], [8, 8]]
    weights: [0.60, 0.25, 0.15]
  num_boxes:
    values: [1, 2]
    weights: [0.9, 0.1]
  topology_steps:
    mode: relative_choices
    multipliers: [0.75, 1.0, 1.25, 1.5]
    weights: [0.15, 0.50, 0.25, 0.10]
  p_change_directions:
    values: [0.15, 0.35, 0.60, 0.80]
    weights: [0.15, 0.50, 0.25, 0.10]
  reverse_search_depth:
    values: [100, 300, 500]
    weights: [0.20, 0.60, 0.20]
  reverse_progress:
    values: [near_goal, middle, far]
    weights: [0.20, 0.55, 0.25]
  wall_ratio_range: [0.0, 0.55]
```

第一版 split id 固定为 train=0、validation=1、test=2，并令 `generation_seed = dataset_seed + split_id * 10_000_000 + task_index`，因此示例中 train 的 task index 0 对应 18427。每次候选构造使用 `numpy.random.SeedSequence([generation_seed, attempt_index])` 派生 PCG64 RNG；生成代码不得混用未显式播种的全局 NumPy/Python RNG。dataset seed、NumPy 版本和 bit-generator 名称必须写入审计报告。

`dim_room=[H,W]` 表示棋盘有 `H` 行、`W` 列。第一版明确排除 6x8、8x6 等非正方形尺寸，避免同时引入长宽比变化。manifest 中出现的 `dim_room: [6,6]`、`topology_steps: 20`、`reverse_search_depth: 300` 等只是某一条任务的已解析结果，用来重放该任务；下一条任务可以采到完全不同的组合。这些隐藏生成字段不进入模型 prompt。

#### `topology_steps` 的当前实际基线与兼容语义

`generate_room(..., num_steps=25)` 中的 `25` 只是函数签名默认值，并不是 VAGEN 默认 6x6 地图的实际取值。当前调用链为：

```text
gym_sokoban 根据地图尺寸计算 self.num_gen_steps
    -> RagenSokobanEngine.reset()
    -> generate_room(num_steps=self.num_gen_steps)
```

基础计算为：

```text
auto_topology_steps = floor(1.7 * (H + W))
```

因此 6x6 地图的当前默认值是：

```text
floor(1.7 * (6 + 6)) = 20
```

为了保持现有训练和验证行为兼容，同时允许数据生成器扩大拓扑分布，第一版固定增加下面的配置语义：

```yaml
# null 表示完全保持当前 gym_sokoban 自动计算行为
topology_steps: null

# 仅供数据集生成器使用；与固定 topology_steps 二选一
topology_steps_sampling:
  mode: relative_choices
  multipliers: [0.75, 1.0, 1.25, 1.5]
  weights: [0.15, 0.50, 0.25, 0.10]
```

对 6x6 基准值 20，解析后的候选值为：

```text
15, 20, 25, 30
```

同时必须支持整数范围模式：

```yaml
topology_steps_sampling:
  mode: integer_range
  min: 14
  max: 30
```

配置解析规则必须唯一且可审计：

1. `topology_steps` 为整数时，所有任务固定使用该值；
2. `topology_steps` 和 `topology_steps_sampling` 同时出现时报错，避免静默覆盖；
3. 两者都未配置时，使用当前 `floor(1.7 * (H + W))` 行为；
4. 采样模式先根据地图尺寸计算自动基准，再将 `auto_topology_steps * multiplier` 按 decimal half-up 规则取到最近整数（恰好 `.5` 时向远离 0 的方向取整）；禁止直接使用 Python `round()` 的 ties-to-even 行为；
5. manifest 始终记录最终解析出的 `topology_steps`，不能只记录 multiplier；
6. 生成器版本、采样策略和随机 seed 一起决定结果，确保普通 GAE 训练、验证和离线重放使用同一任务真值。

同时必须把 `p_change_directions` 暴露到新实验的 `SokobanStateGeneratorConfig` 和复制后的 `RagenSokobanEngine`。它目前虽然是 `generate_room()` 的参数，但原上层没有显式传入，所以实际一直使用函数默认值 `0.35`。第一版固定独立采样：

```yaml
p_change_directions_sampling:
  values: [0.15, 0.35, 0.60, 0.80]
  weights: [0.15, 0.50, 0.25, 0.10]
```

`topology_steps` 控制随机游走/开辟地板的次数，`p_change_directions` 控制随机游走改变方向的概率。两者共同决定墙和地板的拓扑，不能用其中一个替代另一个。生成后仍应以实际 `wall_ratio`、连通区域大小、走廊/分叉统计做分桶和验收，因为参数与最终拓扑复杂度不是严格一一对应。

本文的 `wall_ratio` 统一指 `interior_wall_ratio = 内部墙格数 / ((H-2)*(W-2))`，不包含必然为墙的外边界。`corridor_cell_ratio` 的分母是内部非墙格，分子是其中恰好只有两个相反方向非墙邻居的格子数；分母为 0 的地图直接判非法。manifest 和 audit 只能使用这两个定义，不能混入按全部 `H*W` 格子计算的另一种 wall ratio。

`sampled_reverse_depth` 是主要多样性来源。必须从反向生成过程中得到的多个合法、可解状态中按分布采样，而不是总取搜索结束时的最后一个状态。它能够直接产生远离目标、中等进度和接近完成的状态。

当前 RAGEN `reverse_playing()` 只返回得分最高的最终状态、box mapping 和 action sequence，并不保留可直接采样的中间状态。因此这不是“给现有函数多传一个参数”即可完成，复制后的实验引擎必须增加有界快照收集器：

```text
DFS 每发现一个新的合法 reverse state
    -> 计算 reverse depth、box displacement、box swaps 和 state hash
    -> 按 near_goal / middle / far 候选桶做 reservoir sampling
    -> 每个桶只保留固定上限的状态副本，禁止保存全部 explored states
    -> 生成结束后从目标桶抽一个候选
    -> 用前向 Sokoban 引擎验证该候选可解并记录 sampled_reverse_depth
```

快照先按真实 reverse depth 分层：每个 depth 使用独立 reservoir，最多保留 16 个候选；搜索结束后令 `d_max` 为实际达到的最大非零深度，并按 `1..ceil(d_max/3)`、`ceil(d_max/3)+1..ceil(2*d_max/3)`、其余深度分别定义 `near_goal/middle/far`。从各深度 reservoir 合并结果中，再按 seed 决定的稳定随机 rank 为每个最终桶最多保留 256 个候选。目标桶为空时整条任务重采样，不降级到其他桶。manifest 必须记录候选的真实 reverse depth、`d_max`、bucket、状态 hash 和验证得到的最短解长度。

#### 玩家有时贴箱子、有时远离

`player_reposition_steps` 只能限制随机移动的最大步数，不能可靠控制玩家最终与箱子的距离。数据集生成器必须使用显式的玩家位置分桶，直接从同一可达区域内选择满足距离条件的空地或目标格。

距离应定义为玩家在不推动箱子的条件下，到最近箱子相邻可站立格的最短路径长度，而不是简单的曼哈顿距离。墙体会让曼哈顿距离产生误导。

第一版固定分布：

```yaml
player_placement_sampling:
  distance_metric: shortest_walk_distance_to_box_contact
  buckets:
    adjacent:
      range: [0, 0]
      weight: 0.30
    near:
      range: [1, 2]
      weight: 0.30
    medium:
      range: [3, 5]
      weight: 0.25
    far:
      range: [6, null]
      weight: 0.15
```

这里 `adjacent=0` 表示玩家已经站在箱子的一个相邻可接触格；`near=1..2` 表示还需走 1-2 步才能接触箱子。

具体生成流程：

1. 先用反向搜索确定墙、目标和箱子位置；
2. 计算不穿墙、不穿箱子的玩家可达区域；
3. 对每个候选玩家格计算到最近箱子可接触格的最短路径距离；
4. 按 `adjacent/near/medium/far` 的欠缺配额选择目标 bucket；
5. 从 bucket 中均匀选择玩家位置；
6. 若目标 bucket 在当前地图没有候选格，则重新生成地图，不允许静默降级到其他 bucket；
7. manifest 记录目标 bucket、实际最短路距离和最终玩家坐标。

`adjacent` 必须进一步细分并统计：

```text
pushable_adjacent：玩家贴着箱子，箱子前方可移动
blocked_adjacent：玩家贴着箱子，但箱子被墙或其他箱子挡住
```

adjacent 样本内部固定为 `pushable_adjacent` 70%、`blocked_adjacent` 30%。这样既包含“开局就在箱子边上”的状态，也包含必须先绕行的状态。环境给定动作的事件平衡仍在后续 query sampler 中独立完成。

### 4.3 当前状态类别配额

生成器先为每条任务抽取一个互斥的 `primary_state_category`，再调用该类别对应的构造/拒绝采样分支。正式训练集按以下比例预先换算成整数任务数，不能让生成结果自然漂移：

| 状态类别 | 比例 |
|---|---:|
| 普通可解状态 | 50% |
| 玩家紧贴箱子 | 20% |
| 接近完成状态 | 15% |
| 至少一个箱子已在目标上 | 5% |
| 玩家位于目标上 | 5% |
| 狭窄走廊或高墙密度 | 5% |

第一版 train、validation 和 test 全部只包含经前向求解器验证可解的状态，上述六类生成路线配额合计 100%。不构造死锁/不可解子集，也不根据最短解长度划分 easy、medium 或 hard；棋盘规模差异只通过 6x6、7x7、8x8 三种尺寸体现。

这里的 100% 是**生成路线配额**，不是所有语义属性的互斥占比。一个以“玩家靠近箱子”分支生成的样本仍可能同时出现 `box_on_target=true`，但其 `primary_state_category` 仍是 `player_near_box`，不会再次占用 `box_on_target` 的主类别配额。每条任务另外记录下列非互斥布尔标签：

```text
is_player_adjacent_box
is_near_completion
has_box_on_target
is_player_on_target
is_narrow_or_high_wall
```

`ordinary_solvable` 分支是唯一例外：候选必须不满足其余五个特殊标签，防止普通类别成为未受控的兜底桶。审计同时报告主类别的精确数量和这些多标签属性的实际边际比例；不能拿边际比例反过来判定主类别配额失败。

六个主类别的接受谓词固定如下，当前状态均必须尚未通关：

| primary category | 必须满足的谓词 |
|---|---|
| `ordinary_solvable` | 不满足下列任何特殊类别谓词 |
| `player_near_box` | 玩家距离桶为 `adjacent`；其中 pushable/blocked 再按 70/30 分层 |
| `near_completion` | `shortest_solution_steps` 为 1-3 |
| `box_on_target` | 至少一个但并非全部箱子位于目标；因此第一版该类别强制 `num_boxes=2` |
| `player_on_target` | `fixed[player_position]=target`；第一版该类别强制 `num_boxes=2` |
| `narrow_or_high_wall` | `interior_wall_ratio >= 0.35` 或 `corridor_cell_ratio >= 0.50`，且不属于 solver unknown |

`num_boxes` 的总体 90%/10% 仍是硬配额：`box_on_target` 与 `player_on_target` 两类各占 5% 的双箱额度，其余主类别固定为单箱。这样既精确保持双箱 10% 的总体比例，也避开难以稳定生成的“双箱 + far”等交叉组合。尺寸、箱子数和主类别先联合生成一张整数 contingency table，再按表构造任务；不能先独立抽样后依靠无限 rejection 碰运气。

本文中的 `wall_ratio` 统一指 `interior_wall_ratio = 内部墙格数 / ((H-2)*(W-2))`，不把必然存在的外边界墙计入；main-v1 只接受 `[0.0,0.55]`。`corridor_cell_ratio` 的分母是内部非墙格，分子是恰好只有两个相反方向非墙邻居的内部格。`box_target_distance` 只作为审计 metadata 记录，不作为第一版独立采样轴，避免增加重复约束。

validation 与 test 使用同一组六类主类别比例，并分别通过 largest-remainder 法换算为自身规模的整数配额。

特殊类别的构造路径固定为：`box_on_target` 只从双箱反向快照中筛选“部分箱子在目标”的状态；`near_completion` 只通过前向 BFS 最短解 1-3 的过滤得到；`player_on_target` 在选定可解快照后，只从玩家同一可达区域内的空目标格选择新位置，再重新做前向可解性验证。不能通过无约束改写数组制造状态。

### 4.4 数据集变量、占比与约束类型总表

本节是 `main-v1` 数据分布的统一入口。实现时不得只读取其中某一张表，也不得把所有变量当成相互独立的随机变量。先为硬配额生成整数 contingency table，再在满足这些配额的前提下做条件采样和拒绝采样。50,000 条训练集对应数量如下；validation 和 test 除尺寸外使用相同比例，并用 largest-remainder 法换算为整数。

#### 4.4.1 地图与生成参数

| 变量 | 取值及占比 | 50,000 条 train 数量 | 约束类型 |
|---|---|---:|---|
| 棋盘尺寸 `dim_room` | 6x6 60%；7x7 25%；8x8 15% | 30,000；12,500；7,500 | 硬配额 |
| 箱子数 `num_boxes` | 1 个 90%；2 个 10% | 45,000；5,000 | 硬配额 |
| 目标数 `num_targets` | 1 个 90%；2 个 10% | 45,000；5,000 | 与箱子数绑定，始终 `num_targets=num_boxes`，不是独立采样轴 |
| `topology_steps` 相对自动基准的倍率 | 0.75 取 15%；1.0 取 50%；1.25 取 25%；1.5 取 10% | 7,500；25,000；12,500；5,000 | 硬配额；先按尺寸计算自动基准，再取整得到实际 step |
| `p_change_directions` | 0.15 取 15%；0.35 取 50%；0.60 取 25%；0.80 取 10% | 7,500；25,000；12,500；5,000 | 硬配额 |
| `reverse_search_depth` 上限 | 100 取 20%；300 取 60%；500 取 20% | 10,000；30,000；10,000 | 硬配额；它是搜索上限，不是最终状态的实际深度 |
| 反向快照阶段 `reverse_progress` | `near_goal` 20%；`middle` 55%；`far` 25% | 10,000；27,500；12,500 | 硬配额；按本次搜索实际 `d_max` 的三等分定义 |
| 旧随机玩家重定位参数 | `player_move_probability=0`、`player_continue_probability=0`、`player_reposition_steps=0` | 全部 50,000 | 固定关闭；改用显式玩家距离桶 |
| 地图形状 | 正方形 100%；非正方形 0% | 50,000；0 | 固定约束 |
| 渲染器与 sprite | 当前 VAGEN Sokoban 原生渲染 100% | 50,000 | 固定约束；不采样主题、噪声或图像增强 |

validation 和 test 为了便于按尺寸比较，不复用 train 的 60%/25%/15%：validation 为 6x6/7x7/8x8 = 667/667/666，test 为 3,334/3,333/3,333。除这一项外，上表比例保持不变。

#### 4.4.2 玩家位置和当前状态

| 变量 | 取值及占比 | 50,000 条 train 数量 | 约束类型 |
|---|---|---:|---|
| 玩家到最近箱子可接触格的最短可行走距离 | `adjacent=0` 30%；`near=1..2` 30%；`medium=3..5` 25%；`far>=6` 15% | 15,000；15,000；12,500；7,500 | 硬配额 |
| `adjacent` 内部类型 | `pushable_adjacent` 70%；`blocked_adjacent` 30% | 占全体 10,500；4,500 | 条件硬配额；只以 15,000 条 adjacent 为分母 |
| 主状态生成路线 `primary_state_category` | 普通可解 50%；玩家紧贴箱子 20%；接近完成 15%；部分箱子在目标上 5%；玩家在目标上 5%；狭窄走廊或高墙密度 5% | 25,000；10,000；7,500；2,500；2,500；2,500 | 互斥硬配额 |
| 可解性 | `solvable` 100%；`unsolvable` 0%；`solver_unknown` 0% | 50,000；0；0 | 发布前硬过滤 |
| 当前状态是否已经通关 | 未通关 100%；已通关 0% | 50,000；0 | 固定约束 |

玩家距离桶和主状态生成路线是两张需要联合满足的表，而不是同一个变量。例如全体有 30% 的 `adjacent` 状态，其中 20% 由“玩家紧贴箱子”主路线产生，剩余 10% 分配到其他主路线；实现应预先生成联合整数配额，不能依赖无界随机碰撞。

尺寸、玩家距离、地图 topology 和主状态路线必须联合分配：`far>=6` 的 15% 全部由 `ordinary_solvable` 承担，并与 8x8 的 15% 配额一一配对；其中 10% 使用 `topology_multiplier=1.5`、5% 使用 1.25，且全部使用 `p_change_directions=0.35`。`narrow_or_high_wall` 的 5% 固定使用 `topology_multiplier=0.75`、`p_change_directions=0.15` 和 `reverse_progress=near_goal`，但不固定玩家距离。`player_on_target` 固定使用 near 距离。原因是小棋盘 far、低 topology far、以及高 topology 与狭窄路线等交叉单元无法在固定 attempt 上限内稳定构造。联合槽位仍保持尺寸、距离、topology、方向变化概率、反向阶段和六类主路线的全部全局硬边际不变；不得直接减少某类样本。

`near_completion` 与反向阶段、玩家距离也联合分配：全部 15% `near_completion` 使用 `reverse_progress=near_goal`；其中要求最后一步通关的 10% 任务使用 `adjacent`，剩余 5% 非终止 near-completion 任务使用 `near`。再加上 `player_near_box` 路线固定占用的 20% adjacent，恰好得到全局 adjacent 30%。这是由最短解 1-3 步的定义推导出的可行性约束：状态若距离箱子接触格尚有 6 步以上，不可能在 1-3 个动作内完成。联合分配后仍保持主路线、玩家距离和反向阶段三组总体边际比例不变。

`has_box_on_target`、`is_player_on_target`、`is_near_completion` 和 `is_narrow_or_high_wall` 可以在非对应主路线中重叠出现，所以它们的**实际边际占比不等于主路线的 5%/15%/5%**。第一版只硬控互斥主路线，不再为这些重叠布尔属性设置第二组互相冲突的硬比例；audit 必须报告它们的实际数量和占比。

#### 4.4.3 动作块与状态转移

| 变量 | 取值及占比 | 50,000 条 train 对应数量 | 约束类型 |
|---|---|---:|---|
| 动作块长度 `k` | 1 步 40%；2 步 35%；3 步 25% | 20,000；17,500；12,500 个任务 | 硬配额 |
| primitive event | 缺口权重：`player_move` 25%；`box_push` 20%；`wall_noop` 15%；`blocked_box_noop` 15%；`box_enters_target` 10%；`box_leaves_target` 5%；`player_target_transition` 10% | 共 92,500 次 primitive transition；实际整数由合法动力学决定并写入 audit | 软采样权重；另以第 5.2 节最低覆盖率硬验收 |
| 动作方向 | 每种 event 内 Up/Down/Left/Right 各约 25% | 不预先给全局固定整数 | 条件平衡；1,000 条 smoke 最大差 10 个百分点，正式集最大差 8 个百分点 |
| no-op 与有效变化 | 软目标约为 no-op 30%、产生状态变化 70% | 实际整数由各 event 合计并写入 audit | 不单独采样；受事件锚点和合法动力学影响 |
| 最后动作是否通关 | 是 10%；否 90% | 5,000；45,000 个任务 | 硬配额；前 `k-1` 个动作通关率必须为 0% |
| 实际执行动作数 | 等于给定动作数 100%；存在未执行后缀 0% | 50,000；0 | 固定验收约束 |
| 监督状态数量 | 每条任务只监督一个动作块结束后的最终状态 100%；监督中间 trajectory 0% | 50,000；0 | 固定任务定义 |

上述 92,500 次 primitive transition 来自动作长度硬配额：`20,000*1 + 17,500*2 + 12,500*3 = 92,500`。no-op 的 30% 是 `wall_noop 15% + blocked_box_noop 15%`；有效变化的 70% 是其余五类事件之和，不能再设置一套独立且可能冲突的采样比例。

#### 4.4.4 有差异但不单独指定占比的派生变量

以下变量仍然构成数据多样性并写入 manifest/audit，但第一版**不设置独立比例**：

| 派生变量 | 第一版处理方式 | 不单独硬控的原因 |
|---|---|---|
| `interior_wall_ratio` | 只接受 `[0.0,0.55]`；audit 按 `[0,0.20)`、`[0.20,0.35)`、`[0.35,0.55]` 报告直方图 | 它由尺寸、`topology_steps` 和转向概率共同决定；再强行独立定额会制造冲突。高墙/窄走廊已由 5% 主路线保证最低覆盖 |
| `corridor_cell_ratio`、分叉数、连通区域大小 | 记录实际值并报告分布 | 都是拓扑生成后的结果，不是独立环境旋钮 |
| `shortest_solution_steps` | 所有样本记录整数并报告直方图；1-3 步由 15% `near_completion` 主路线保证最低覆盖 | 用户已决定不按难度分级，因此不再按解长度建立 easy/medium/hard 配额 |
| `sampled_reverse_depth` 实际整数 | 记录实际 depth、`d_max` 和二者比值 | 已通过 near_goal/middle/far 的 20%/55%/25% 控制阶段，实际整数依搜索树而变 |
| `box_target_distance` | 记录并报告直方图 | 与反向阶段、最短解长度和箱子覆盖目标状态高度相关，第一版不重复约束 |
| `has_box_on_target`、`is_player_on_target` 等多标签属性 | 报告实际边际比例 | 属性可重叠；硬配额以互斥 `primary_state_category` 为准 |
| 整个动作块的最终状态是否恰好不变 | 从 current/next hash 派生并报告 | 多步动作可能先变化再返回，不等同于 primitive no-op 比例 |

不设置独立比例不等于忽略这些变量。构建报告必须给出分 split 的计数、占比和直方图，供训练前检查是否异常集中；只有本节明确标为硬配额或容差目标的项目才决定构建是否通过。

#### 4.4.5 唯一性和 split 隔离

| 变量 | 目标占比 | 约束类型 |
|---|---:|---|
| split 内唯一 `map_hash` | 100% | 硬验收 |
| split 内唯一 `current_state_hash` | 100% | 硬验收 |
| split 内唯一 `task_hash` | 100% | 硬验收 |
| train/validation/test 地图 hash 交集 | 0% | 硬验收 |

第一版不生成非正方形棋盘，也不对网格做 padding；prompt 中的 `{rows}`、`{cols}` 和 parser 的目标 shape 随每条任务的 `dim_room` 改变。若 smoke test 暴露多尺寸图像批处理问题，应修正预处理或按尺寸分桶组 batch，不能静默把 7x7、8x8 从训练集移走。

### 4.5 求解器、资源上限与失败语义

“可解”和 `shortest_solution_steps` 使用同一个确定性前向 BFS 定义：状态节点由玩家位置和全部箱子位置组成，静态 `fixed` 层不变；动作扩展顺序固定为 `Up, Down, Left, Right`；第一次找到全部箱子位于目标的状态时，其深度就是最短解长度。第一版默认资源上限固化在 sampler profile 中：

```yaml
generation_limits:
  max_task_attempts: 500
  max_reverse_nodes: 200000
  max_forward_bfs_nodes: 500000
  max_snapshot_candidates_per_depth: 16
  max_snapshot_candidates_per_bucket: 256
  max_tasks_per_map: 1
  max_queries_per_current_state: 1
```

`reverse_search_depth` 限制单条反向路径深度，`max_reverse_nodes` 限制一次反向搜索累计展开的状态数，两者不是同一个参数。前向 BFS 若在 `max_forward_bfs_nodes` 内找到解则记为 `solvable`；若搜索空间穷尽仍无解则记为 `unsolvable`；触及节点上限则记为 `solver_unknown`。train、validation 和 test 只发布 `solvable`，其余候选全部丢弃，不建立不可解或难度子集。

每个目标任务最多进行 `max_task_attempts` 次候选构造，attempt seed 由 `(generation_seed, attempt_index)` 确定。达到上限仍无法满足指定尺寸、主状态类别、动作事件和求解约束时，整个数据构建命令以非零状态失败，并在 audit 中记录失败 bucket；不得换桶、放宽约束或复制已有任务补填数。wall-clock timeout 只作为进程看门狗，触发后中止构建，不参与样本标签，保证同一版本与 seed 的接受/拒绝由确定性节点上限决定。

第一版每个 `map_hash` 只发布一条任务，每个 `current_state_hash` 也只发布一个动作查询；发现重复 map、current state 或 task 时都丢弃候选并继续同一目标任务的 attempt。因而每个 split 的唯一 `map_hash`、`current_state_hash` 和 `task_hash` 数量必须都等于该 split 的任务数。后续若为降低生成成本而复用地图，必须建立新 schema/profile，不能静默改变第一版的数据独立性。

## 5. 环境给定动作的构造

### 5.1 单轮动作块任务定义

每条任务包含 1-3 个 primitive actions，但模型只预测动作块执行完毕后的一个最终状态：

```text
(RGB_t, [a_t^1, ..., a_t^k]) -> current_state_t, final_state_t+k, k in [1, 3]
```

`query_actions` 中每一项只能是 `Up`、`Down`、`Left`、`Right`。环境按顺序执行，保存动作块执行后的唯一 `next_state`，然后任务结束。训练环境设置 `max_turns: 1`。

这里的“单轮”是指模型只生成一次，“最终状态”是指只监督动作块结束后的状态。数据集不要求输出中间状态，因此没有 `trajectory` 字段。

动作块长度固定为：

| 长度 | 比例 |
|---|---:|
| 1 | 40% |
| 2 | 35% |
| 3 | 25% |

### 5.2 转移事件平衡

不能直接独立、均匀地采样动作块。由于棋盘大部分格子不变，原始随机动作容易产生大量简单 no-op。生成器应执行候选动作块，分类其中每个 primitive transition，再按全局缺口选择样本。

primitive transition 使用下列采样优先权重；它们用于生成时计算全局缺口，不承诺由于动作块动力学耦合而无法稳定实现的精确边际：

| 事件 | 比例 |
|---|---:|
| `player_move` | 25% |
| `box_push`（普通地板之间） | 20% |
| `wall_noop` | 15% |
| `blocked_box_noop` | 15% |
| `box_enters_target` | 10% |
| `box_leaves_target` | 5% |
| `player_target_transition` | 10% |

上表按所有动作块内部的 primitive transitions 统计。为了避免方向偏置，每一种事件内部还要平衡 `Up/Down/Left/Right`。此外设置可审计的最低覆盖率：`player_move>=20%`、`box_push>=12%`、`wall_noop>=10%`、`blocked_box_noop>=8%`、`box_enters_target>=5%`、`box_leaves_target>=2%`、`player_target_transition>=7%`。这些下限合计小于 100%，剩余质量允许由实际棋盘动力学决定；manifest 与 audit 必须报告实际精确比例。

为保证稀有事件不是纸面目标，主状态路线与动作块设置硬锚点：`box_on_target` 必须包含 `box_leaves_target`，且固定使用长度 3、玩家距离 `near`；`player_on_target` 必须包含 `player_target_transition`；`player_near_box` 的 `pushable_adjacent/blocked_adjacent` 必须分别包含成功推箱事件/`blocked_box_noop`；终止任务必须以 `box_enters_target` 结束。找不到锚点动作块时重新生成整条状态。

每个 primitive transition 必须且只能记入一个 `primary_event`。分类按以下优先级执行，命中后停止，避免一次推箱同时占多个桶：

```text
1. box_enters_target
2. box_leaves_target
3. box_push
4. blocked_box_noop
5. player_target_transition
6. wall_noop
7. player_move
```

其中 `box_push` 只指既不进入也不离开目标的成功推动；推动目标上的箱子时，即使玩家随后站上该目标，也只记 `box_leaves_target`。`blocked_box_noop` 指玩家试图推箱但箱后格不可进入；`wall_noop` 只指直接撞墙。`player_target_transition` 合并玩家进入和离开目标两种情况，并在 `event_subtype` 中保留 `enters_target/leaves_target`；玩家成功移动且没有发生更高优先级事件时，才进入第 5 或第 7 类。

约束优先级固定为：中间动作不得通关和最终终止约束 > 动作块长度配额 > 稀有事件硬锚点 > primary event 缺口权重 > 事件内方向平衡。正式 50,000 条训练集的主状态类别、动作块长度和最终通关任务数使用 largest-remainder 法预分配为精确整数。primitive event 通过上述最低覆盖率验收；事件样本数至少 100 时，四方向最大占比与最小占比之差在 1,000 条 smoke audit 中不得超过 10 个百分点，在不少于 10,000 条的正式 audit 中不得超过 8 个百分点。未达到覆盖率或方向容差则数据构建失败，不得发布 manifest。

### 5.3 终止语义

训练数据不允许出现“动作块在第 i 步已经通关，但 `query_actions` 后面仍有动作”的样本。当前 VAGEN 的提前 `break` 可作为在线执行的容错行为，但数据生成阶段应消除这种未执行动作后缀，避免 prompt 与监督目标之间存在歧义。

构造长度为 `k` 的动作块时，必须在隐藏环境副本中逐步模拟：

```text
for i in 1..k:
    枚举并试执行候选动作
    if i < k:
        过滤所有会使关卡完成的候选动作
    if i == k:
        根据 terminal 配额决定是否允许完成关卡
```

验收规则：

- 当前状态本身必须未通关；
- 第 `1..k-1` 个动作执行后必须仍未通关；
- 只有第 `k` 个动作允许完成关卡；
- 若没有满足目标事件和终止约束的动作，整条候选任务重新采样，不能保留未执行后缀，也不能静默缩短动作块；
- `executed_action_count` 必须始终等于 `len(query_actions)`；
- `terminated=true` 当且仅当最后一个动作完成关卡；
- `next_state` 始终是完整执行所有 `query_actions` 后的状态；
- 模型仍只输出一个最终 `<prediction>`，状态建模奖励不额外支付通关奖励。

正式 train、validation 和 test 中固定 10% 的任务在最后一个动作恰好通关，其余 90% 执行动作块后保持非终止；数量通过 largest-remainder 法预分配。该比例必须写入 dataset sampler profile，不能由随机生成自然形成。

## 6. 图像输入与渲染一致性

第一版只使用当前 VAGEN Sokoban 的原生渲染器和 sprite，不增加渲染主题、替代 sprite set、颜色扰动、压缩伪影、模糊、噪声、缩放、padding、旋转或翻转。训练时看到的 RGB 分布应与后续 VAGEN Sokoban 下游环境保持一致。

数据多样性只由环境语义产生：

- 6x6、7x7、8x8 棋盘尺寸；
- 多种房间拓扑和墙体密度；
- 玩家、箱子、目标的多种相对布局；
- 箱子/玩家覆盖目标时由原渲染器产生的复合 sprite；
- 不同反向搜索阶段和玩家距离桶；
- 1-3 个动作及不同 transition event。

manifest 不保存 PNG 或图片路径。每次 `reset` 必须把 manifest 中标准化的 `current_state.fixed + current_state.entity` 交给同一 VAGEN renderer，确定性生成 RGB；manifest 固化 `renderer_version`，版本不匹配时拒绝读取。这里的目标不是训练视觉域泛化，而是训练与当前 VAGEN 图像域严格对齐的状态提取和转移预测。训练前审计负责对每条状态调用渲染器并检查 RGB dtype、尺寸和通道数。

## 7. RL Prompt 数据集与环境接口

第一版固定新增独立环境 `SokobanStateModeling`，不直接复用让模型自主选择动作的 `Sokoban.step()` 协议。

### 7.1 Reset

```text
reset(seed):
    将 VAGEN 传入的 seed 解释为 manifest task_index
    通过 manifest 字节偏移索引读取唯一任务
    加载该任务已保存的 current_state 和 next_state
    用固定 renderer 从 current_state 现场生成 RGB
    读取长度为 1-3 的 query_actions
    返回 RGB + query_actions + 输出协议
```

在线 rollout 以 manifest 中已经固化的 `current_state` 和 `next_state` 为唯一奖励真值，不在每次 reset 时重新生成地图或重新计算监督目标。动作重放一致性已在数据构建审计阶段强制验证；环境启动时可对每个 manifest 做一次版本/hash 检查，debug 或离线评估模式可以再次重放抽查，但正常训练 reset 只读取真值，避免每个 rollout 重复求解并防止引擎版本变化后标签静默漂移。

实际送给模型的 prompt 必须显式包含符号表、动作规则、给定动作块和严格输出格式，不能只给 RGB 与动作列表。第一版固定使用如下零样本模板，其中 `{rows}`、`{cols}` 和 `{actions_json}` 由环境填充：

```text
You are predicting Sokoban state transitions from an RGB image.

The image shows the current {rows}x{cols} board. Use exactly one character per cell:
# = wall
_ = empty floor
O = empty target (uppercase letter O)
X = box on floor
P = player on floor
* = box on target
+ = player on target

The environment executes these actions in order: {actions_json}
Action semantics:
- Up/Down/Left/Right attempts to move the player by one cell.
- Moving into a wall has no effect.
- Moving into a box pushes it by one cell only if the cell behind it is free floor or a target; otherwise the action has no effect.
- Every listed action is executed. If the episode terminates, it can only happen on the final listed action.

Infer the current symbolic board from the image, then predict the single final board after all actions.
Return no explanation. Output exactly {rows} grid rows between each pair of tags, with exactly {cols} symbols per row. Use this exact tag order:
<perception>
...grid rows...
</perception>
<prediction>
...grid rows...
</prediction>
```

`{rows}` 和 `{cols}` 使用十进制整数；`{actions_json}` 使用无换行 JSON 数组，例如 `["Right", "Right", "Down"]`。动作名大小写固定，不随机改写成箭头、数字或同义词。

模板在整个训练集内保持一致即可；多样性应来自地图、状态、动作块和渲染，而不是改写符号定义。为了防止模型只记忆模板，reward parser 必须独立检查标签、行列数、字符集合以及两个网格的状态合法性。

### 7.2 Step

```text
step(model_response):
    解析 <perception> 与 <prediction>
    检查两个网格的尺寸和字符表
    将复合网格解码成 fixed + entity
    与隐藏 current_state / next_state 比较
    返回一个终局标量 reward
    done = True
```

同一个 `task_id` 在训练重试、验证和离线重放时必须共享完全相同的：

```text
generation_seed
resolved_generator_config
由 current_state 和 renderer_version 唯一确定的 current RGB
query_actions
next_state
```

生成器的随机数必须只由任务 seed 决定。不能在同一个 `task_id` 重放时重新随机生成状态或动作，否则 reward 和评估结果不可复现。

### 7.3 Manifest 与 VAGEN Dataset 接入

当前 `vagen/training/dataset.py` 只向环境传递 `env_name + seed + config + max_turns`，不会直接传入 manifest 行。第一版不新增训练 Dataset 类，而是把 VAGEN 的 `seed` 明确定义为当前 split manifest 的整数 `task_index`：

```yaml
envs:
  - name: SokobanStateModeling
    n_envs: 50000
    seed: [0, 49999, 1]
    max_turns: 1
    response_length_per_turn: 512
    stop_strings: ["</prediction>"]
    config:
      manifest_path: /absolute/shared/path/train.jsonl
      manifest_index_path: /absolute/shared/path/train.idx
```

`seed: [0,49999,1]` 在现有 `AgenticDataset` 中表示从闭区间抽取 50,000 个互不重复的整数，恰好形成全部 task index 的一次随机排列。训练和验证使用各自 manifest，因此可以分别从 0 开始编号。

数据集生成时同时写出 JSONL 和 `.idx` 字节偏移表。环境构造时只打开 manifest store；`reset(seed)` 根据 index 定位并读取一行，不能每个 rollout 重新解析 50,000 行。manifest offset index 使用进程内只读缓存，以绝对规范化路径为 cache key；Ray worker 间不共享 Python 对象，但所有 worker 必须能访问配置中的同一个绝对路径。生成过程每 100 条原子提交一个 staging chunk；中断后使用同一 split、总量、版本和 sampler profile 加 `--resume` 恢复，最多重算当前尚未提交的 chunk。正式 manifest 与 index 发布后删除 staging chunk。

每个 split 的 `task_index` 必须等于该任务在 JSONL 中从 0 开始的行号，连续且无重复；`.idx` 第 i 项就是第 i 行首字节偏移，条目数必须等于 manifest 行数。启动检查发现不连续 index、越界 offset 或 `.idx`/JSONL 数量不一致时直接拒绝训练。

manifest store 初始化时强制检查 schema/generator/renderer version、`.idx` 条目数和首尾 offset；每次训练 reset 强制校验 task index、数组 shape 和 `current_state_hash/next_state_hash`。RGB 不落盘，由环境在 reset 中使用当前声明版本的 renderer 从 `current_state` 生成，因此不存在图片路径或图片 SHA-256。

### 7.4 训练集完整遍历

不能仅依赖当前普通 `RandomSampler + drop_last` 的 epoch 行为来声称“50,000 条都会在重复前出现”：`50,000` 不能被 batch size `128` 整除，epoch 尾部可能被丢弃。第一版新增确定性的 `FullCoverageSampler`，配置为：

```yaml
data:
  train_batch_size: 128
  shuffle: true
  sampler:
    class_path: sokoban_state_modeling.training.sampler.FullCoverageSampler
    full_coverage_cycle: true
    pad_final_batch: true
```

每个 coverage cycle 先由训练 seed 生成全部 50,000 个 dataset row index 的唯一排列，再顺序 yield 全部唯一 index；最后才依次重复该排列开头的 index，使总数成为 batch size 的整数倍。对 50,000/128，一轮为 391 个完整 batch、50,048 次 rollout，其中最后追加排列开头的 48 个 index。下一轮使用 `seed + cycle_id` 生成新的唯一排列。这样前 50,000 个 sampler 位置没有重复，任何唯一任务也不会被 `drop_last` 丢掉；48 个 padding rollout 单独记录为 `coverage_padding_rollouts`，不计入“唯一任务覆盖数”。

sampler 必须实现 `state_dict/load_state_dict`，保存 cycle、排列 RNG 状态和当前位置，checkpoint 恢复后不能从 cycle 开头重放。若 `pad_final_batch=false`，启动时直接报错，因为当前 PPO 更新要求固定 batch；不能静默回退到普通 RandomSampler。

### 7.5 Parser 的唯一边界

parser 先把 `CRLF` 规范化为 `LF`，除此之外不改写模型文本，不对棋盘行调用 `strip()`。每个 section 独立提取，因此一段出错不阻止另一段获得状态分数：

```text
section_syntax_valid(name) =
    对应 opening/closing tag 各恰好出现一次
    and opening 在 closing 之前
    and tag 内恰好 H 行
    and 每行恰好 W 个字符
    and 每个字符都属于 # _ O X P * +
```

重复标签、缺失标签、嵌套标签、空行、行首/行尾空格、错误行列数或非法字符都使对应 section 无效，该 section 的所有状态奖励贡献为 0。错误标签顺序或标签外的额外文本不妨碍两个 section 分别提取和计算状态分数，但会使整体格式分为 0。

`format_valid=1` 只要求两个 section 均可独立解码且 `<perception>` 位于 `<prediction>` 之前；它是 parser 调试信号，允许标签外存在额外文本，不作为 11 项正式指标或 reward。`format_exact`（dense 中的 `format_score` 及正式指标 `format_exact_rate` 与它使用同一布尔量）进一步要求输出完全符合模板：opening/closing tag 各占一行、tag 与网格之间没有空行、没有缩进或尾随空格、两个 section 之间没有额外行、首字符就是 `<perception>`，并且 `</prediction>` 后最多只有一个换行符。任何解释、Markdown fence 或思维过程都使其为 0。

syntax valid 只表示网格可以解码，不代表预测状态符合 Sokoban 约束。dense reward 仍对可解码网格计算逐类 F1，从而保留部分学习信号。语义合法状态必须恰好有 1 个玩家、箱子数和目标数与任务配置一致、任何 entity 都不在 wall 上，且外边界全部为 wall；否则该 section 的 exact 必为 0，也不能用于 `transition_consistency_rate` 的引擎装载。玩家完全被困仍是可装载的合法状态，只是可能不可解，不能仅因没有有效移动而判编码非法。预测的墙/目标位置可以与真值不同而仍然语义合法，此时可计算部分分和内部转移一致性，但 exact 仍为 0。`exact_only` 不使用这些部分分。

## 8. 可验证奖励设计

### 8.1 防止“复制当前状态”刷分

一个 1-3 步动作块通常仍只改变少数格子。如果只计算全棋盘 cell accuracy，模型直接复制当前棋盘也可能获得很高分。因此最终状态必须同时评价：

- 完整棋盘；
- 玩家/箱子/目标实体集合；
- 相对上一真实状态发生变化的格子及其新旧类别；
- 整盘完全正确。

基础分数定义：

```text
perception_full  = 0.4 * perception_fixed_macro_f1
                 + 0.6 * perception_entity_macro_f1
perception_exact = 1 if the entire perceived two-layer board is exact else 0

prediction_full  = 0.4 * prediction_fixed_macro_f1
                 + 0.6 * prediction_entity_macro_f1
delta_score      = F1 over exact (row, col, old_type, new_type) change atoms
prediction_exact = 1 if the entire predicted two-layer board is exact else 0

format_score = 1 if the response strictly follows the two-tag grid protocol else 0
exact_all    = perception_exact * prediction_exact * format_score
```

逐类 F1 在每条任务内部计算。`fixed_macro_f1` 是 wall、floor、target 三个二值 mask F1 的平均；`entity_macro_f1` 是 player、box、target 三个对象 mask F1 的平均，其中 target mask 来自 fixed 层，叠加符号按第 3.2 节展开。单类采用 `2TP/(2TP+FP+FN)`；若预测与真值对该类都为空则该类 F1 定义为 1，只有一侧为空则为 0。虽然正常真值包含这些必要类别，仍固定空集规则以覆盖非法预测和测试。

对于执行完整个动作块后真实状态不变的样本，`delta_score` 定义为：只有预测状态与当前状态完全一致时得 1，否则得 0。不能因为真实变化集合为空而自动给满分。

### 8.2 两版 Reward Profile

#### A. `dense`：部分正确可得分

这版用于提供连续学习信号，但通过较高 exact 权重限制“差不多正确”获得的分数：

```text
perception_score = 0.50 * perception_full
                 + 0.50 * perception_exact

prediction_score = 0.25 * prediction_full
                 + 0.25 * delta_score
                 + 0.50 * prediction_exact

reward = 0.40 * perception_score
       + 0.50 * prediction_score
       + 0.05 * format_score
       + 0.05 * exact_all
```

所有项都裁剪到 `[0,1]`。若某个标签中的网格无法合法解析，对应的 perception 或 prediction 子分数为 0；另一段若可合法解析，仍可独立获得部分分。`format_score` 只有严格满足标签顺序、数量、行列数、字符表且无额外可见文本时才为 1。

这一版中，不完整正确的单个 perception 或 prediction 子任务最高只能获得其子分数的 0.5；整盘 exact 才能把该子分数提升到 1。训练 success 指标仍只按整盘 exact match 统计，不能用连续 reward 代替成功率。

#### B. `exact_only`：三部分独立 exact 奖励

这版完全不使用 F1、cell accuracy 或 delta 部分分，只使用三个二值 exact 判断：

```text
format_exact     = 1 if the complete output protocol is exact else 0
perception_exact = 1 if the entire perceived board is exact else 0
prediction_exact = 1 if the entire predicted board is exact else 0

reward = 0.10 * format_exact
       + 0.40 * perception_exact
       + 0.50 * prediction_exact
```

三个部分独立计分：格式正确可得 0.1，`<perception>` 整盘完全正确可得 0.4，`<prediction>` 整盘完全正确可得 0.5。某个棋盘只要错一个格子，该棋盘对应项就是 0，不按正确格子数量给部分分；但另一部分如果 exact，仍可获得其对应奖励。因此该版 reward 可能为 0、0.1、0.4、0.5、0.6、0.9 或 1.0，而不再是只有 0/1。

11 项正式指标仍照常计算并写入日志，但 F1、cell accuracy、changed-cell F1 和 transition consistency 全部只是诊断信息，不参与 `exact_only` reward。条件成功率的 numerator/denominator 只作为内部聚合项保存在 rollout metadata 中，不作为第 12、13 个正式指标展示；`reward_metric_error` 同样属于基础设施健康诊断。

两版实验必须使用相同的模型初始化、train/validation manifest、任务顺序、batch size、训练 steps 和随机 seed，只改变 `reward.profile`：

```yaml
reward:
  profile: dense       # 主实验

# 消融实验改为：
# reward:
#   profile: exact_only
```

普通 GAE 要求 batch 内存在 reward/advantage 方差。B 版初期可以先从 0.1 的格式项学习协议；但当 batch 内格式都正确后，如果 perception 和 prediction 的 exact 命中仍长期为 0，状态相关 reward 就没有方差，模型无法继续学习状态能力。这应被如实报告为稀疏状态奖励失败，不能偷偷加入 F1 或按格子部分分来改变 `exact_only` 定义。

奖励不使用 LLM judge，不使用自然语言关系抽取，也不包含 Sokoban 通关奖励。它只衡量视觉状态提取和动作条件状态预测。

### 8.3 成功率与诊断指标

训练 reward 是连续分数；报告中的“成功率”使用严格的整盘 exact match，不通过人为阈值定义成功。先将 `<perception>` 和 `<prediction>` 的复合字符网格解码为标准 `fixed + entity`，再与隐藏真值逐格比较：

```text
perception_success_i = 1
    iff perception section parses to a valid grid
    and decoded perception_i == current_state_i

prediction_success_i = 1
    iff prediction section parses to a valid grid
    and decoded prediction_i == next_state_i

joint_success_i = perception_success_i * prediction_success_i
```

这里的 `joint_success_i` 只衡量两个状态是否都正确；即使标签外有额外可见文本，只要两段仍能合法解析，它也不把格式错误混入状态能力指标。环境交给 VAGEN 的 episode 成功标志采用更严格口径：`info["success"] = bool(exact_all)`，因此现有 `traj_success` 只有在格式、perception 和 prediction 三者都 exact 时才为 1。`joint_success_rate` 与 `traj_success` 必须分别命名和记录，不能混用。

数据集级指标为：

```text
perception_success_rate = sum(perception_success_i) / N
prediction_success_rate = sum(prediction_success_i) / N
joint_success_rate      = sum(joint_success_i) / N

conditional_prediction_success_rate
    = sum(joint_success_i) / sum(perception_success_i)
```

其中 `conditional_prediction_success_rate` 只在 `<perception>` 完全正确的样本上计算 `<prediction>` 成功率，更接近模型在“已经正确看懂当前状态”之后的纯转移建模能力。若分母为 0，则记为 `N/A`，不能记为 0。

该条件指标不能作为每条样本的普通 0/1 值再直接求均值。环境应额外返回两个内部聚合量 `conditional_prediction_numerator=joint_success_i` 与 `conditional_prediction_denominator=perception_success_i`，训练/验证聚合器和离线评估器统一计算 `sum(numerator) / sum(denominator)`。这两个只是计算辅助量，不计入下面的 11 个模型能力指标。

评估报告固定记录以下 11 个模型能力指标：

| # | 指标名 | 定义 |
|---:|---|---|
| 1 | `perception_success_rate` | `<perception>` 与隐藏 `current_state` 整盘 exact match 的样本比例。 |
| 2 | `prediction_success_rate` | `<prediction>` 与隐藏 `next_state` 整盘 exact match 的样本比例。 |
| 3 | `joint_success_rate` | perception 与 prediction 同时整盘完全正确的样本比例。 |
| 4 | `conditional_prediction_success_rate` | 在 perception 完全正确的子集中，prediction 完全正确的比例；分母为 0 时记为 `N/A`。 |
| 5 | `format_exact_rate` | 完全满足 reward 所用严格双标签网格协议的样本比例；与 `format_score` 使用同一判定。 |
| 6 | `perception_cell_accuracy` | perception 中 `(fixed, entity)` 同时正确的格子数占全部格子的比例。 |
| 7 | `prediction_cell_accuracy` | prediction 中 `(fixed, entity)` 同时正确的格子数占全部格子的比例。 |
| 8 | `perception_entity_macro_f1` | perception 的 player、box、target 三类对象掩码 F1 的宏平均；`*` 同时计入 box 和 target，`+` 同时计入 player 和 target。 |
| 9 | `prediction_entity_macro_f1` | prediction 的 player、box、target 三类对象掩码 F1 的宏平均，叠加符号处理同上。 |
| 10 | `changed_cell_f1` | 预测变化原子 `(row, col, old_type, new_type)` 与真实变化原子集合之间的 F1。 |
| 11 | `transition_consistency_rate` | 从模型 `<perception>` 出发，由真实引擎执行 `query_actions` 后恰好得到模型 `<prediction>` 的样本比例。 |

第 6-11 项属于诊断指标，不应通过人为阈值重新命名为“成功率”。`transition_consistency_rate` 只能诊断模型内部转移是否自洽，不能替代对隐藏真值的 perception/prediction 成功率，因为模型可能在错误的初始棋盘上得到自洽但仍错误的预测。

模型仍然只输出完整棋盘，`changed_cell_f1` 不要求它额外输出“变化列表”：评估器分别比较真实的 `current_state -> next_state` 与模型的 `<perception> -> <prediction>`，自动派生两组变化原子；其中 `old_type/new_type` 是该格的规范复合类型 `(#, _, O, X, P, *, +)`。`transition_consistency_rate` 则把合法解析的 `<perception>` 装载进同一引擎并执行给定动作块；任一网格非法或引擎无法装载时该样本记 0。

上述 11 项还要按动作块长度、transition event、地图尺寸、箱子数和终止状态做分桶切片；切片不计为新的指标。优化器使用的 `mean_reward` 单独作为训练过程统计，也不计入这 11 个模型能力指标。

数据集级 cell accuracy、entity macro F1、changed-cell F1 和 transition consistency 均先得到每条任务的分数，再对任务做等权宏平均，不把所有格子汇总后做微平均。因此 8x8 样本不会仅因格子更多而获得更高权重；按尺寸切片后仍使用同一规则。success/rate 指标按对应二值分子除以有效任务数，条件成功率除外，它使用前述独立 denominator。

模型正常返回但严格格式或棋盘解析失败时，这是模型错误：逐样本 `format_exact_rate=0`，相应 exact 和连续诊断贡献记 0。只有环境崩溃、manifest 损坏或奖励代码异常等基础设施错误，才将该行可直接计算的指标贡献及条件指标辅助量填为 `NaN` 并另增 error counter；不能用 `NaN` 掩盖模型的非法输出。表中的 `*_rate` 是这些逐样本贡献经过数据集聚合后的名称。

## 9. 数据规模与拆分

第一阶段数据规模固定为：

```text
train: 50,000 tasks
validation: 2,000 tasks
test: 10,000 tasks
```

50,000 条训练任务按尺寸分配为：6x6 30,000 条、7x7 12,500 条、8x8 7,500 条。该规模的目的不是增加优化步数，而是在固定训练预算下减少 prompt 重复，提高地图、状态、动作块和事件的覆盖率。

validation 和 test 不沿用训练集的 60/25/15 尺寸比例，而是在 6x6、7x7、8x8 间等量分层，以便直接按尺寸比较。固定尺寸顺序为 6x6、7x7、8x8：validation 为 667/667/666，test 为 3,334/3,333/3,333。二者使用与 train 相同的生成参数支持集、六类状态路线、动作块长度、primary event 和 10% 最终通关配额，但拥有完全独立的 `map_hash`。评估只报告总体和按棋盘尺寸切片的结果，不再设置 clean/hard 或 easy/medium/hard 子集。

第一版两个 reward profile 的训练预算固定为：

```text
train_batch_size: 128
total_training_steps: 400
rollout.n: 1
总 episode rollouts: 128 * 400 = 51,200
```

数据加载器使用第 7.4 节的完整覆盖 cycle。按上述预算，前 391 steps 共执行 50,048 个 rollout，其中包含全部 50,000 个唯一任务和 cycle 尾部 48 个 padding 重复；剩余 9 steps 的 1,152 个 rollout 来自重新打乱后的下一 cycle。过滤、失败重试、coverage padding 和分布式补 batch 必须分别计数，不能把它们静默算成新的唯一任务。

只有在增加训练步数、需要更多唯一任务覆盖且当前指标仍未饱和时，才把训练集扩展到 200,000 tasks。若训练预算仍只有约 51,200 个 rollouts，直接生成 200,000 条会导致大部分任务从未进入训练，因此不扩容。扩容时优先增加唯一状态和事件组合，而不是重复同一状态的不同随机动作。

拆分必须基于规范化地图哈希，而不是随机拆分 JSONL 行。四个哈希分别定义为：

```text
map_hash
    = SHA256(schema_version || H || W || canonical_fixed_bytes)

current_state_hash
    = SHA256(schema_version || map_hash || canonical_entity_bytes)

next_state_hash
    = SHA256(schema_version || map_hash || canonical_next_entity_bytes)

task_hash
    = SHA256(schema_version || current_state_hash || canonical_query_actions)
```

`map_hash` 只包含棋盘尺寸及 `fixed` 层，即墙、地板和目标；绝不能包含玩家、箱子或所谓 initial placement，否则同一静态地图上的不同状态会被误分到不同 split。`current_state_hash` 在地图上加入当前实体摆放；`task_hash` 再加入有序动作块，用于区分同一状态上的不同查询。规范化序列化必须包含 schema/version、shape、dtype 和明确的长度边界，不能直接拼接无分隔文本；`next_state_hash` 单独记录，不参与 split 归属。

要求：

- 将 SHA-256 digest 的前 8 个 byte 按 big-endian 解为无符号整数，令 `u = value / 2^64`，并固定归属：`[0,0.61)` 为 train、`[0.61,0.73)` 为 validation、`[0.73,1)` 为 test；区间宽度约按正式 split 大小的平方根分配，以降低总拒绝采样成本且不让 validation 只拥有极窄的 hash 空间。各 split 生成器只接受落入自身区间的地图，直到该 split 的任务配额精确填满；
- 先按上述 `map_hash` 规则决定 split，再从该地图生成状态和动作；
- 同一地图的所有直接生成状态只属于一个 split；
- 同一状态及其不同动作查询只属于一个 split；
- train/validation/test 不共享 `map_hash`，因此也不得共享 `current_state_hash`；
- 同一 split 内用 `task_hash` 去除完全重复任务；
- train、validation 和 test 使用同一生成 profile，只通过 map hash 隔离。

可以复用项目现有的 map hash partition 思路，但数据集应将 split 结果固化在 manifest 中，避免生成器升级后样本集合静默变化。

## 10. 数据生成后的审计

生成过程必须产出审计报告，至少包含：

```text
任务总数及各 split 数量
唯一 map_hash / current_state_hash / next_state_hash / task_hash 数量
重复状态比例
query action 数量 1/2/3 的分布
所有 primitive actions 的四方向分布
各 transition event 分布
各 primary_state_category 和 state_attributes 分布
地图尺寸和箱子数分布
最短解长度分布
solvable / unsolvable / solver_unknown 候选数量；发布任务必须全部为 solvable
wall ratio 分布
box_on_target / player_on_target 出现率
no-op 比例
最后一个动作完成关卡的比例
中间动作提前通关违规数
每个 split 的哈希交集
动作块最终状态重放失败数
现场渲染 RGB 的 dtype、尺寸或通道错误数
FullCoverageSampler 每 cycle 的 unique / padding / duplicate 数量
```

强制验收条件：

1. 所有状态满足 Sokoban 编码约束；
2. 每个状态恰好一个玩家；
3. 箱子数和目标数符合任务配置；
4. 保存的 `next_state` 能够由引擎顺序执行 `query_actions` 重放得到；
5. 所有 current state 均能由声明版本的 renderer 生成尺寸正确的 uint8 RGB；
6. train/validation/test 的 `map_hash`/`current_state_hash` 无泄漏，且各 split 内无重复 `task_hash`；
7. 主类别、尺寸、动作块长度和终止数的精确整数配额完全匹配；primitive event 全部达到第 5.2 节最低覆盖率，事件内方向差满足 smoke/正式集对应容差；
8. 对 manifest 的每条任务构造规范 golden response，两种 reward profile 都必须得到 reward=1，11 项可适用指标均为满分；
9. 在 `next_state != current_state` 子集上，“复制当前棋盘作为 prediction”的 `prediction_exact=0`、`changed_cell_f1=0`，且 dense `prediction_score <= 0.25`；真实 no-change 子集单独报告，复制得到满分属于正确行为；
10. 全 `_` 常量棋盘基线在两个 profile 的 `joint_success_rate=0`，exact_only reward 只能得到格式项 0.10；dense 会按设计给局部匹配支付连续分数，因此只报告其实际均值，不设置与连续奖励定义矛盾的 0.10 阈值。“最常见动作效果”基线同样报告但不设脱离数据分布的武断阈值；
11. 每条任务都满足 `executed_action_count == len(query_actions)`；
12. 长度为 `k` 的动作块在前 `k-1` 个动作后均未通关，若任务终止，只能由第 `k` 个动作触发；
13. train、validation 和 test 发布的任务全部为 `solvable`，`unsolvable` 与 `solver_unknown` 发布数量均为 0；
14. parser 逐项测试缺失/重复/乱序标签、额外文本、Markdown fence、CRLF、空行、行空格、非法字符和错误尺寸，结果必须符合第 7.5 节；
15. `FullCoverageSampler` 测试必须证明前 50,000 个位置无重复、cycle 长度为 50,048、尾部恰为 48 个 padding index，并且在任意位置保存/恢复后，后续 index 序列逐项一致。

## 11. 实施顺序

### Phase A：生成器与状态快照

- 增加 `dim_room` 采样，第一版只包含 6x6、7x7、8x8；先抽取尺寸，再计算该尺寸的 topology 自动基准，不生成非正方形棋盘；
- 为新实验的 `SokobanStateGeneratorConfig` 和复制后的 `RagenSokobanEngine` 增加可选 `topology_steps`；`null` 时保持 gym_sokoban 按尺寸计算的默认值；
- 在数据生成器层增加基于自动值的 `relative_choices` 和 `integer_range` 采样策略，并在 manifest 固化解析后的整数；
- 显式透传并支持采样 `p_change_directions`，不再只能使用 `generate_room()` 的 0.35 默认值；
- 为复制后的 RAGEN 生成器开放反向深度和玩家重定位参数，并实现按进度桶有界保存、去重和采样反向搜索快照；
- main-v1 将旧随机玩家重定位参数固定关闭，由显式距离桶放置玩家；
- 支持直接从反向生成轨迹的不同深度采样当前状态；
- 实现规范化 `fixed + entity` 快照和 state hash；
- 实现状态合法性和可解性验证，不生成难度标签；
- 实现基于最短可行走距离的玩家位置分桶，并分别控制贴箱子、近距离、中距离和远距离样本；
- 对贴箱子状态进一步标记 `pushable_adjacent` 与 `blocked_adjacent`。

### Phase B：动作块查询与隐藏真值

- 实现四方向后继枚举和 transition event 分类；
- 实现按事件缺口采样动作；
- 每条任务先采样动作块长度 `k in {1,2,3}`，再在隐藏环境副本中逐个构造 `query_actions`；
- 构造前 `k-1` 个动作时过滤所有会直接通关的候选，最后一个动作是否通关由 terminal 配额决定；若中途无合法候选，则整条任务重采样；
- 完整执行动作块，只保存最终 `next_state` 作为预测目标，不保留任何通关后不会执行的动作后缀；
- 额外记录 `event_sequence`、`event_subtype_sequence`、`executed_action_count` 和 `terminated` 用于数据审计，不要求 VLM 输出中间状态。

### Phase C：物化数据与审计

- 输出 JSONL manifest、`.idx` 字节偏移索引和 generator/renderer version；训练时现场渲染 RGB，不物化 PNG；
- 审计报告固化 dataset seed、完整 sampler profile、Python/NumPy/PCG64 版本、gym-sokoban 版本、七张 sprite 的 SHA-256，以及最终 manifest/index SHA-256；
- 按地图哈希先拆分，再生成状态与增强；
- 实现并测试完整覆盖 sampler，50,000/128 的每个 cycle 必须得到 50,000 unique + 48 padding；
- 生成统计和泄漏审计报告；
- 人工可视检查随机样本及全部稀有事件类别。

### Phase D：RL 环境与奖励

- 新增 `SokobanStateModeling` 单轮环境；
- 实现严格的 `<perception>/<prediction>` parser；
- 实现 perception、prediction、full、delta、exact 和 format 分数，并支持 `dense` 与 `exact_only` 两个可配置 reward profile；
- 将各子分数组合成一个终局标量 reward，使用 verl 原生 `algorithm.adv_estimator=gae` 并启用 critic，不启用 `default_gae`、`bi_level_gae`、token-level GAE 或分段 credit assignment；本实验每行就是一个完整的单轮 episode，不需要 VAGEN 的跨行轨迹拼接；
- 增加复制状态、常量状态、完美预测和错一步预测的单元测试；两种 profile 必须分别断言其预期 reward。

### Phase E：小规模验证后扩容

- 先生成 1,000 条 smoke 数据；
- 用完美答案确认 reward=1；
- 用复制当前状态基线检查防刷分效果；
- 运行小规模 Qwen3.5-4B PPO + 原生 `gae`，启用 critic，检查 advantage、value loss、reward variance 和格式收敛；
- 通过后生成 50,000 条正式训练任务；
- 根据失败类型扩充稀有状态和事件，而不是盲目增加均匀随机样本。

## 12. 第一版明确不做的事情

- 不让 VLM 自主选择动作；
- 不使用 `prefix_actions` 构造或复现当前状态；
- 不把通关成功作为训练奖励；
- 不用 LLM judge 解析预测；
- 不只监督对象相对玩家的方向关系；
- 不输出或监督中间 trajectory；每条任务执行 1-3 个环境动作后只预测一个最终状态；
- 不允许生成器版本变化后继续复用未校验的旧 manifest；
- 不以普通 cell accuracy 作为唯一指标或唯一奖励。

## 13. 实验目录与文件改造计划

### 13.1 总体边界

在 VAGEN 仓库根目录直接新建 `sokoban_state_modeling/`，不增加 `experiments/` 中间层。状态建模实验的环境、生成器、状态编码、prompt、奖励、指标、配置、脚本和测试全部收拢在该目录中。

原有 `Sokoban` 环境继续用于自主选动作和多轮通关训练，不改变其任务语义。主项目只增加环境注册、通用指标透传/聚合和可选 sampler 工厂接入；第一版不修改 `vagen/envs/sokoban/sokoban_env.py`、公共 response format 或公共 state reward。

### 13.2 需要新建的目录

```text
sokoban_state_modeling/
├── __init__.py
├── README.md
├── DATASET_PLAN.md
├── env/
│   ├── __init__.py
│   └── sokoban_state_env.py
├── engine/
│   ├── __init__.py
│   ├── env.py
│   └── utils.py
├── state/
│   ├── __init__.py
│   ├── codec.py
│   ├── transition.py
│   └── validation.py
├── generator/
│   ├── __init__.py
│   ├── sampler.py
│   ├── action_sampler.py
│   ├── dataset_builder.py
│   ├── manifest_store.py
│   └── audit.py
├── prompt/
│   ├── __init__.py
│   ├── template.py
│   └── parser.py
├── reward/
│   ├── __init__.py
│   ├── reward.py
│   └── metrics.py
├── training/
│   ├── __init__.py
│   └── sampler.py
├── configs/
│   ├── train_qwen35_4b.yaml
│   ├── eval_qwen35_4b.yaml
│   └── test_qwen35_4b.yaml
├── scripts/
│   ├── generate_dataset.py
│   ├── train_qwen35_4b.sh
│   └── evaluate.py
├── tests/
│   ├── test_codec.py
│   ├── test_generator.py
│   ├── test_action_sampler.py
│   ├── test_prompt_parser.py
│   ├── test_reward_metrics.py
│   ├── test_full_coverage_sampler.py
│   └── test_env_smoke.py
├── data/                    # 运行时生成，不提交 Git
│   ├── manifests/
│   └── audits/
└── outputs/                 # checkpoint、日志和评估结果，不提交 Git
```

实施开始时，将当前根目录的 `SOKOBAN_VLM_RL_DATASET_PLAN.md` 移动并重命名为 `sokoban_state_modeling/DATASET_PLAN.md`，后续只维护新位置，避免存在两个内容逐渐分叉的计划文件。

### 13.3 新文件职责

| 文件 | 职责 |
|---|---|
| `README.md` | 实验目标、安装方法、数据生成、训练和评估命令。 |
| `DATASET_PLAN.md` | 本文档，作为数据格式、奖励和验收标准的唯一来源。 |
| `env/sokoban_state_env.py` | 实现注册名为 `SokobanStateModeling` 的单轮环境；reset 返回 RGB、动作块和协议，step 解析回答、评分并立即结束；每次 step 在 `info["reward_metrics"]` 中返回固定的逐样本指标及条件成功率的 numerator/denominator。 |
| `engine/env.py` | 实验专用 Sokoban 引擎封装，开放地图和反向生成参数，支持状态加载、克隆、重放与渲染。 |
| `engine/utils.py` | 实验专用房间生成和反向搜索逻辑。 |
| `state/codec.py` | `fixed + entity`、复合字符网格与 hash 的双向转换。 |
| `state/transition.py` | 顺序执行 1-3 个 primitive actions，返回最终状态、事件序列和终止信息。 |
| `state/validation.py` | 检查尺寸、玩家数、箱子/目标数、状态合法性、可解性和重放一致性；train、validation、test 全部强制可解。 |
| `generator/sampler.py` | 从 dataset-level profile 采样地图尺寸、箱子数、拓扑、反向深度和玩家距离桶，并固化 resolved config。 |
| `generator/action_sampler.py` | 构造 1-3 动作块、平衡事件与方向，并保证只有最后一个动作可以通关。 |
| `generator/dataset_builder.py` | 先按 `map_hash` 固化 split，再物化 JSONL manifest、四类 hash 和 generator/renderer version；不写 PNG。 |
| `generator/manifest_store.py` | 使用 `.idx` 字节偏移按 `task_index` 随机读取 manifest；提供按绝对路径键控的进程内只读缓存，避免每个 rollout 全量解析 JSONL。 |
| `generator/audit.py` | 生成分布、重复、泄漏、动作重放、提前终止和图像一致性审计报告。 |
| `prompt/template.py` | 生成包含符号表、动作规则、动作块和 `<perception>/<prediction>` 输出协议的零样本 prompt。 |
| `prompt/parser.py` | 严格解析两个标签、行列数和字符集合，不解析自然语言关系。 |
| `reward/reward.py` | 计算 perception、prediction、changed-cell、exact 和 format 子分数，并按配置选择 `dense` 或 `exact_only` 终局标量奖励。 |
| `reward/metrics.py` | 计算并聚合固定的 11 个模型能力指标及分桶切片。 |
| `training/sampler.py` | 实现可 checkpoint 恢复的 `FullCoverageSampler`：唯一排列优先、cycle 尾部补齐固定 batch，并发布覆盖与 padding 计数。 |
| `configs/*.yaml` | 独立的训练/评估数据路径、采样分布、奖励权重和 Qwen3.5-4B 原生 `gae` 配置。 |
| `scripts/generate_dataset.py` | 数据生成命令入口，先支持 smoke 数据，再支持正式规模。 |
| `scripts/audit_splits.py` | 在三个 split 都发布后检查 `map_hash/current_state_hash/task_hash` 的两两交集。 |
| `scripts/train_qwen35_4b.sh` | 启动单轮 RL 训练，不调用原 Sokoban 多轮 action prompt。 |
| `scripts/evaluate.py` | 加载固定 test manifest，默认要求所有 task index 各出现一次且不可缺失/重复，输出 11 项总体及分桶指标；`--allow-partial` 仅用于明确标注的开发报告。 |
| `tests/*.py` | 覆盖编码往返、确定性生成、动作终止约束、解析、奖励、指标和环境端到端 smoke test。 |

### 13.4 从现有 VAGEN 复制或复用的代码

| 现有来源 | 新位置 | 处理方式 |
|---|---|---|
| `vagen/envs/sokoban/ragen_engine/env.py` | `sokoban_state_modeling/engine/env.py` | 复制后保留来源注释，再开放本实验需要的生成参数、状态克隆和重放接口；不反向修改原文件。 |
| `vagen/envs/sokoban/ragen_engine/utils.py` | `sokoban_state_modeling/engine/utils.py` | 复制反向生成逻辑后参数化深度、拓扑和玩家重定位；保留确定性 seed。 |
| `vagen/envs/sokoban/sokoban_env.py` | `sokoban_state_modeling/env/sokoban_state_env.py` | 只复制环境生命周期、图像渲染和 seed/map partition 骨架；删除自主动作解析、多轮循环和通关奖励。 |
| `vagen/envs/sokoban/utils/prompt.py` | `sokoban_state_modeling/prompt/template.py` | 参考 VAGEN 的 `<perception>/<prediction>` 术语，不复制 few-shot、`<reasoning>` 或 `<answer>` 协议。 |
| `vagen/envs/_common/response_format.py` | `sokoban_state_modeling/prompt/parser.py` | 复用标签提取思路，但实现本任务严格的双网格 parser，避免改变公共四段式协议。 |
| `vagen/envs/_common/rewards/state.py` | `sokoban_state_modeling/reward/` | 参考奖励配置和日志接入方式；棋盘评分改为确定性的两层状态比较，并只向原生 GAE 返回一个终局标量 reward。 |
| `examples/train/sokoban/*.yaml`、Qwen3.5 启动脚本 | `sokoban_state_modeling/configs/`、`scripts/` | 复制训练框架参数骨架，替换环境名、单轮长度、prompt 长度、数据路径和奖励配置。 |

`vagen/training/`、`verl/`、`GymImageEnv` 和训练基础设施不复制到实验目录，直接从现有 VAGEN 导入使用，避免维护一份完整框架副本。

这里复用 VAGEN 是复用多模态 rollout、环境适配、数据装载、PPO/critic、分布式训练、日志和 checkpoint 基础设施，不代表使用 VAGEN 的 bi-level GAE。新实验的算法配置固定为：

```yaml
algorithm:
  adv_estimator: gae
  gamma: 1.0
  lam: 1.0

critic:
  enable: true

trainer:
  harness: concat

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: vanilla
  rollout:
    n: 1
```

`max_turns: 1` 表示每个 episode 只允许一次模型回答；它不限制 prompt 内 `query_actions` 的长度，后者仍为 1-3。环境第一次 `step(response)` 必须主动返回 `done=True`，TurnLimit 只是防止实现错误产生第二轮。必须覆盖 `AgenticDataset` 默认的 `stop_strings: ["</answer>"]`，改用 `</prediction>`，否则本任务没有 `<answer>` 时可能生成到 token 上限。

### 13.5 必须修改的现有项目文件

为注册新环境、接入完整覆盖 sampler 并让 11 项指标以正确口径进入训练日志，第一版需要修改五个现有代码库文件：

| 文件 | 修改内容 |
|---|---|
| `vagen/configs/env_registry.yaml` | 新增 `SokobanStateModeling: sokoban_state_modeling.env.sokoban_state_env.SokobanStateModelingEnv`。 |
| `vagen/training/main.py` | `create_rl_sampler()` 支持配置指定的 sampler `class_path`，只在显式配置时实例化 `FullCoverageSampler`；其他训练继续使用现有 Random/SequentialSampler。 |
| `vagen/envs/_common/adapter.py` | 增加通用 `reward_metrics` 收集接口；根据环境声明的固定指标名初始化键集合。模型输出非法时保留环境给出的 0 分指标；只有环境/评分异常时才返回同一组键并填 `NaN`、同时记录 error counter，避免 batch 行之间 key set 不一致。 |
| `vagen/training/agent_loop/gym_loop.py` | 将 adapter 收集的固定 `reward_metrics` 合并进 `reward_extra_info`，从而进入 train/validation 日志；不得改变现有环境没有声明该能力时的行为。 |
| `vagen/training/trainer/mixin.py` | 对本环境的 conditional numerator/denominator 做分子和分母分别求和，再发布 `conditional_prediction_success_rate`；不能对逐行占位值直接求均值。同时读取 sampler 状态并记录 coverage unique/padding/cycle 计数。验证和离线评估采用同一条件比率公式。 |

`setup.py` 当前使用 `find_packages()`；只要新增目录包含上述 `__init__.py`，新的顶层 Python package 会被自动发现，因此第一版不需要修改 `setup.py`。若以后要求 wheel 内携带实验 YAML，再单独将 `sokoban_state_modeling/configs/*.yaml` 加入 package data。

### 13.6 明确不修改的现有文件

- `vagen/envs/sokoban/sokoban_env.py`：保留原有多轮决策实验；
- `vagen/envs/sokoban/ragen_engine/*`：实验使用复制后的独立版本；
- `vagen/envs/_common/response_format.py`：不把双标签协议强塞进公共 parser；
- `vagen/envs/_common/rewards/state.py`：不改变现有自然语言状态奖励；
- 现有 `examples/train/sokoban/*`：不覆盖原训练配置和脚本；
- `verl/*`：不改上游优化器、GAE 或 rollout 基础设施；
- `vagen/training/*` 中除 `main.py` 的可选 sampler 工厂、`agent_loop/gym_loop.py` 的通用指标透传和 `trainer/mixin.py` 的条件比率聚合外均不修改；
- `vagen/envs/_common/*` 中除 `adapter.py` 的通用指标收集外均不修改。

新环境自行解析 `<perception>/<prediction>`，并通过 EnvSpec 的 `stop_strings: ["</prediction>"]` 结束生成，因此无需让公共 response parser 接受无 `<answer>` 的双标签协议。

### 13.7 实施批次与完成条件

1. **目录骨架与编码**：建立 package、移动本文档、完成 codec/validation；编码往返和非法状态测试通过。
2. **生成器与动作块**：完成参数采样、状态生成、1-3 动作构造和审计；中间动作提前通关违规数为 0，全部任务可重放。
3. **单轮环境与 prompt**：完成 RGB reset、双标签 parser 和一步结束；完美输出能被稳定解析。
4. **奖励与 11 项指标**：完成确定性评分、分桶聚合及防复制基线；完美答案满分，错误答案按预期降分。
5. **训练接入**：注册新环境，添加 Qwen3.5-4B 配置和脚本；原 Sokoban 测试不回归。
6. **逐级放量**：先运行单元测试和少量内存任务，再生成 1,000 条 smoke 数据，最后在审计通过后生成 50,000 条正式数据并启动 RL。
