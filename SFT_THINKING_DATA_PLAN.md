# Thinking SFT 冷启动数据构造计划

## 1. 目标与边界

本计划为 `thinking_prompt_open=true` 的 Qwen3.5 Sokoban 状态模型构造可审计的
SFT 冷启动数据。目标不是让模型学习自由形式的长篇思维链，而是先用短小、确定、
可由环境逐步重放的 scratchpad 学会：

1. 从 RGB 棋盘恢复玩家、箱子和目标点；
2. 按顺序执行 1--3 个动作；
3. 区分移动、推箱、撞墙、箱子阻塞和目标点转换；
4. 在唯一的 `</think>` 后输出严格的 `<perception>/<prediction>` 网格。

程序化 scratchpad 只作为本次冷启动脚手架。本计划采用一次混合 SFT，不把 1-step、
多步或不同尺寸拆成多个训练阶段；模型自产轨迹回收属于后续增强项，不属于本次正式
12k 冷启动集。

## 2. 不变量

- 当前状态、动作、事件和下一状态只以现有纯函数状态转移器为真值。
- 坐标统一为 1-based `(row,column)`，所有集合按行列排序。
- 不使用教师模型生成事实；推理轨迹由逐步执行结果确定性生成。
- 输入 prompt 与 RL rollout 使用同一个 `build_task_prompt()`。
- chat template 已经打开 `<think>`，因此监督 target **不得**包含 `<think>`，必须包含且
  只包含一个 `</think>`。
- `</think>` 后只能是严格答案；答案必须被现有 parser 接受，并得到满分规则奖励。
- 导出时重新执行所有动作。事件、事件子类型、终止标记或最终状态有任一不一致即失败。
- 不覆盖已有导出文件，先写 `.partial`，完成审计后原子发布。

## 3. 推理语法 v1

轨迹保持机器可读和简短，不生成不可验证的解释：

```text
Initial: player=(4,2); boxes=[(3,3)]; targets=[(2,3)].
Given action 1/1 = Up: front=(3,2)=floor; result=move; player=(4,2)->(3,2); event=player_move.
Final: player=(3,2); boxes=[(3,3)]; solved=false.
```

推箱步骤额外记录 `beyond` 和箱子位移：

```text
Given action 1/1 = Right: front=(2,3)=box; beyond=(2,4)=target; result=push; player=(2,2)->(2,3); box=(2,3)->(2,4); event=box_enters_target.
```

no-op 明确记录原因且不伪造位移：

```text
Given action 1/1 = Up: front=(1,2)=wall; result=noop; event=wall_noop.
```

允许的 cell 标签为 `outside/wall/floor/target/player/player_on_target/box/box_on_target`；
允许的 result 为 `move/push/noop`；event 必须来自现有转移器。

## 4. 导出记录契约

每行 JSONL 包含：

- `sft_schema_version`：固定为 `sokoban-thinking-sft-v1`；
- 源任务身份和 split；
- system/user messages，其中 user prompt 保留 `<image>` 占位符；
- `current_state`，供训练 Dataset 按现有 renderer 动态生成 RGB 图像；
- `reasoning`：不带 thinking 标签的推理正文；
- `answer`：严格的 perception/prediction；
- `target`：`reasoning + "</think>" + answer`；`</think>` 后不能插入空白，
  因为严格答案 parser 要求响应从 `<perception>` 开始；
- 动作、事件、hash 和用于切片评测的 metadata。

不把图片复制到导出目录，避免数据膨胀和 renderer 漂移。训练加载器必须从
`current_state` 调用同一个 `render_state()`。

## 5. 冷启动数据规范（正式版）

### 5.1 单阶段冷启动

本次只做一次混合 SFT：在同一个训练集中覆盖不同尺寸、动作长度、箱子数和转移事件，
训练 1--2 epoch 后进入同分布 PPO/GRPO。可以用正式训练集的小前缀做 smoke test，
但这不算独立训练阶段，也不另建一套分布。

正式 SFT 不包含 5x5；5x5 只保留为诊断 probe。正式 SFT 与后续 RL 的主分布均为
6x6--8x8。

### 5.2 数据规模与精确边际配额

| split | 总数 | 6x6 | 7x7 | 8x8 | 1 action | 2 actions | 3 actions | 1 box | 2 boxes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 12,000 | 7,200 | 3,000 | 1,800 | 4,800 | 4,200 | 3,000 | 10,800 | 1,200 |
| validation | 1,200 | 400 | 400 | 400 | 480 | 420 | 300 | 1,080 | 120 |
| test | 3,000 | 1,000 | 1,000 | 1,000 | 1,200 | 1,050 | 750 | 2,700 | 300 |

对应比例：

- train 尺寸：6x6 / 7x7 / 8x8 = 60% / 25% / 15%；
- validation、test 尺寸：三种尺寸各占 1/3；
- 动作长度：1 / 2 / 3 = 40% / 35% / 25%；
- 箱子数量：1 / 2 = 90% / 10%；
- terminal：false / true = 90% / 10%。

每个样本中的动作由任务提供，模型不生成动作。若样本有两个动作，推理文本必须依次写
`Given action 1/2 = ...` 和 `Given action 2/2 = ...`，并从上一步状态继续执行。

### 5.3 其余生成轴

训练集使用 `MAIN_V1_PROFILE` 的下列边际比例；validation/test 除尺寸改为均匀分布外，
其余保持一致：

| 生成轴 | 取值与比例 |
|---|---|
| topology multiplier | 0.75 / 1.0 / 1.25 / 1.5 = 15% / 50% / 25% / 10% |
| change-direction probability | 0.15 / 0.35 / 0.60 / 0.80 = 15% / 50% / 25% / 10% |
| reverse-search depth | 100 / 300 / 500 = 20% / 60% / 20% |
| reverse progress | near_goal / middle / far = 20% / 55% / 25% |
| player distance | adjacent / near / medium / far = 30% / 30% / 25% / 15% |
| primary state category | ordinary_solvable / player_near_box / near_completion / box_on_target / player_on_target / narrow_or_high_wall = 50% / 20% / 15% / 5% / 5% / 5% |
| adjacent subtype | pushable / blocked = 70% / 30%，只在 `player_distance=adjacent` 时定义 |

七类事件必须全部覆盖：`player_move`、`box_push`、`wall_noop`、
`blocked_box_noop`、`box_enters_target`、`box_leaves_target`、
`player_target_transition`（内部再区分进入/离开目标）。事件目标权重依次为
25% / 20% / 15% / 15% / 10% / 5% / 10%。

一个多动作样本可能贡献多个事件，因此事件比例是全局 deficit-balancing 的目标，不是
“每类样本数严格等于该百分比”的独立配额；最终以 manifest audit 的实际事件计数为准。

### 5.4 比例在代码中如何控制

尺寸、动作长度、箱子数等边际比例不是对每条样本独立随机抽签后期待大数收敛，而是先
转换成整数配额，再分配给样本：

1. `SamplerProfile` 声明每个轴的取值和权重；
2. `QuotaSchedule(total, profile, seed, split)` 为每个轴建立长度为 `total` 的配额列；
3. `largest_remainder(total, weights)` 先计算
   `floor(total * weight / sum(weights))`，再按小数余数从大到小补齐剩余名额，保证每列
   总数严格等于 split 总数；
4. 每个值按整数配额重复，再用 split seed 确定性 shuffle；
5. `DatasetBuilder` 按样本 index 读取整行 assignment，并持续重试地图/状态生成，直到
   候选满足 assignment；被拒绝的候选不会消耗配额；
6. `audit_manifest` 用同一个 `QuotaSchedule` 重建期望计数并逐轴核对；
   `audit_splits` 再检查不同 split 间的地图和样本 hash 泄漏。

例如 train 的尺寸权重 60/25/15 在 12,000 条上会直接得到
7,200/3,000/1,800，而不是“大约”得到这些数。

边际轴之间存在物理约束，生成器会在保持最终边际计数的前提下做耦合修正：

- `box_on_target` 固定为 3 actions，并配 2 boxes、较近玩家；
- `player_on_target` 配 2 boxes、较近玩家；
- `player_near_box` 固定为 adjacent；
- terminal 样本从 `near_completion` 中选取，并配 `near_goal`、adjacent；
- far 玩家距离优先配 8x8；
- `narrow_or_high_wall` 与 topology/direction 参数联动；
- adjacent 样本再按 70/30 分成 pushable/blocked。

动作事件采用另一层控制：`ActionSampler` 枚举指定长度的所有候选动作序列，模拟每一步
事件，再按当前全局事件缺口、动作方向缺口和主类别的稀有事件要求打分，从最高分候选中
用 seed 做确定性 tie-break。它保证覆盖并把事件分布拉向目标权重，但不承诺事件计数像
尺寸、动作长度和箱子数那样逐条精确。

### 5.5 可解性保证与边界

- 初始状态来自反向生成；
- 写入 manifest 前还会经过独立的 forward BFS 求解验证，因此 `source_state` 保证可解；
- 给定动作序列由环境真实执行，故 `next_state` 保证转移合法、标签正确；
- 当前实现没有对 `next_state` 再执行一次求解验证，所以不能声称执行动作后的状态仍全部
  可解。若训练定义要求终态也必须可解，需要在构造器中加入第二次
  `solve_forward(block.next_state)` 过滤并写入对应 metadata。

map split 必须在派生 SFT 记录前完成；同一个 `map_hash` 不得跨 split。

### 5.6 与旧 VAGEN 在线 RL 数据的边界

不要把旧 VAGEN 的在线环境 seed 配置直接当作本项目的离线 SFT manifest。旧实验默认是
固定 6x6、1 box、`min_solution_steps=[1,5]`、`search_depth=300`、`max_turns=5`，
而且动作由策略模型选择；当前 standalone 任务是 6x6--8x8 混合、1/2 boxes、动作由
任务提供、模型预测执行动作后的状态。两者只能作为参数参考，不能混成同一种监督样本。

## 6. 训练拼接与 loss mask

训练 Dataset 应按以下方式拼接，而不是盲目依赖通用多轮 Dataset：

```text
prompt_ids = apply_chat_template(system + user, add_generation_prompt=true,
                                 enable_thinking=true)
target_ids = tokenize(reasoning + "</think>" + answer + eos)
```

- prompt、图像和 padding token：loss mask 0；
- reasoning、`</think>`、answer 和 eos：loss mask 1；
- `truncation=error`；
- 必须检查解码后的 prompt 恰好打开一次 thinking，且 target 不会再次打开 thinking。

如果实现分段加权 loss，程序化 reasoning 权重建议为 0.5，严格 answer 权重为 1.0；
第一版可先统一权重，但必须保持 scratchpad 简短，避免其 token 数压倒最终答案。

## 7. 发布前验证门槛

代码级门槛：

- 每种 transition event 至少一个单元测试；
- 多步动作逐步状态一致；
- 篡改 event/next state/terminated 的记录必须被拒绝；
- target thinking 边界能被 `split_reasoning(..., prompt_open=true)` 唯一解析；
- 每条 answer 的 `score_response(..., profile="exact_only") == 1.0`；
- 相同 manifest 重复导出字节完全一致；
- 完整相关 pytest 通过。

数据级门槛：

- 导出行数与源 manifest 一致；
- `task_id/task_index/map_hash` 保持不变；
- 事件计数写入 audit；
- 源 manifest 和导出文件均记录 SHA-256；
- train/validation/test 继续通过现有 split leakage audit。

## 8. 实施顺序

1. 已完成：纯函数 reasoning trace builder 和严格 SFT record builder；
2. 已完成：manifest-to-JSONL exporter 及事件/协议测试；
3. 已完成：动态 renderer 的 VERL MultiTurnSFTDataset 适配层；
4. 已完成：正式配额/事件/泄漏 preflight 和 Qwen3.5-4B 全参数 SFT 启动脚本；
5. 在服务器生成正式 manifest：train 12,000、validation 1,200、test 3,000；
6. 分别运行 manifest audit，并运行 split leakage audit；
7. 导出 thinking SFT JSONL，并运行正式 SFT preflight；
8. 用 `data.train_max_samples` 对正式 train 前缀做 smoke SFT 和格式评测；
9. 通过后对完整 12k 训练集做一次全参数 SFT（1--2 epoch，默认 1 epoch）；
10. 在 6x6/7x7/8x8 validation、test 上评估，再进入同分布 PPO/GRPO。
