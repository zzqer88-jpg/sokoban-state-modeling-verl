# Sokoban：视觉状态、棋盘预测与可替换规划器

## 当前生效范围（2026-09-24）

详细数据与模型方案见 [SOKOBAN_JEV_VISUAL_SFT_PLAN.md](SOKOBAN_JEV_VISUAL_SFT_PLAN.md)。

当前只规划 **棋盘图片 → 当前每格状态** 的监督微调（SFT），不启动训练。下文 v2 中的动力学、动作策略、求解器与 planner 均作为后续路线存档，不属于当前实施范围。

推荐接口是：完整棋盘图片、棋盘尺寸和坐标约定作为输入；每个坐标对应一个 Jev choice 问题，各自输出七类状态的概率与选择：墙、空地、目标、箱子、目标上的箱子、玩家、目标上的玩家。字符只是输出编码，不设置字符棋盘作为模型输入。每格的七类独立归一化，不能将全棋盘所有候选做一次 softmax。

模块责任：

1. 数据生成器：从引擎同一状态生成 RGB 与逐格真值；真值只作为监督标签，不能进入用户提示。按地图划分训练、验证和测试，再在各集合每个内部生成局面；旧动作数据需对重复当前局面去重。
2. 多模态输入链路：传递真实图片内容，使用模型配套 processor，保留视觉编码与语言模型连接。仅在文本中插入图片路径不构成视觉输入。
3. Jev 问题编译：为每格构造坐标问题和固定语义类别，并保存类别与实际答案 token 的映射。具体请求字段、单 token 可用性、多问题限制、共享图像前缀及批处理方式须以 Jev 服务源码为准。
4. SFT：推荐沿用多模态模型原有输出头，输出每格七类候选概率，用 KL(target || prediction) 监督；精确引擎标签转换为 one-hot 目标分布，不采用单标签的全词表 token SFT。训练问题格式、候选映射与推理时编译结果必须一致。全参数或 LoRA 留待实现与资源决策；这不同于原 Pev 冻结骨干后训练动作打分器的路线。
5. 输出与评估：按坐标组装状态矩阵，同时保留原始概率。报告整盘完全正确率、逐格准确率和各类召回率，单独检查箱子/玩家覆盖目标的格子。结构合法率另报，不能用后处理修复后的分数替代原始识别分数。

本地接入现状：Sokoban 的 `sft/verl_dataset.py` 已能把局面渲染为图片并交给 VERL 多模态数据管线，但目前标签包含 thinking、当前状态和未来状态，需要另设 perception-only 数据配置与目标。Pev 的 `OpenJevTeacher` 目前发送纯文本 state；其 `FrozenQwenBackend` 只保留 language_model，不能直接承担图像编码。工作区没有 Jev 服务本体，因此尚不能断言其服务端完全不支持视觉，也不能给出已验证的服务端补丁。

推进顺序：先确认 Jev 多模态请求与问题编译接口；再定义 perception-only 数据及训练/推理一致的标签格式；随后规划 SFT 和逐格评估。若先做逐行生成整盘字符的常规视觉 SFT，应明确它只是基线，不能据此宣称完成 Jev 并行逐格输出。多格问题可以批处理，但共享视觉计算的实际效果需要实现与测量，不能预设只编码一次图片。

---

日期：2026-09-22。修订：v2，依据用户最新范围重写。状态：研究规划，未实施、未启动训练。

依据：本地 Sokoban 项目 60a1916、Pev 项目 54e5b03，以及文末列出的原论文和作者代码。下文设计、数据规模、预算与推进条件都是建议，不是本项目已经取得的结果。

## 1. 范围与推荐决策

最终只保留三项能力：

1. **当前棋盘是什么**：当前完整 RGB → 每格地形与实体状态。
2. **行动后棋盘会怎样**：当前状态 + 候选动作/动作序列 → 下一棋盘或未来棋盘。
3. **如何完成任务**：根据当前状态与目标，通过代码求解器、独立模型或总 planner 产生计划并执行。

不再设置独立的字符棋盘解析任务，也不再设置“初始图片 + 全部历史动作 → 当前状态”的专门训练阶段。字符/JSON/tensor 只是内部状态接口，历史记录只服务于日志、搜索去重、计划复用和可选误差诊断。

**推荐实施主线：视觉状态模型 + 已验证代码 solver，先完成闭环；独立训练棋盘预测模块，并检验它对规划的贡献；之后比较独立神经 planner 与调用工具的总 planner。** 统一端到端模型作为后续受控对照，不要求第一版所有能力都由一个大模型完成。

这分成两个可以并行推进的目标：

- **工程目标**：可靠地从图片完成 Sokoban。已知规则 solver 可以直接使用，不需要等待学习到的转移模型成熟。
- **研究目标**：棋盘预测学得是否准确，以及它能否在固定预算下改善动作选择或搜索。必须有真的使用预测结果的实验，不能因系统有一个 prediction head 就宣称 world model 在起作用。

## 2. 为什么不需要单独的历史状态追踪

当前项目的规则是确定性的。完整状态可以写为：

s = (walls, goals, boxes, player)

若以剩余时间为有限时域任务的一部分，再加入公开的 remaining_steps。在这些量相同、规则相同的条件下，相同动作产生相同后继状态；当前 apply_action 实现也只依赖当前 StateSnapshot 和 action。

因此，**完整当前棋盘已经足够描述决策所需的环境状态**。从哪条路线走到这里，不改变下一步能否推动箱子。这是本项目规则与输入协议下的判断，不是说任意视觉环境都不需要记忆。

必须区分三件事：

| 内容 | 是否保留 | 原因 |
|---|---|---|
| 通过过去观测补齐当前不可见状态 | 主任务不需要 | 默认完整当前图像，覆盖物使用可辨识的组合 sprite |
| 搜索器的 visited/frontier/计划游标 | 保留 | 节省重复搜索、避免循环；属于规划计算状态 |
| 在同一当前棋盘上做更多内部计算 | 可研究 | recurrent thinking 可能帮助规划，不等同于回忆旧棋盘 |

Guez 等在 Sokoban 上研究了 recurrent computation 和额外 thinking time 的作用。这支持将“当前状态上的内部迭代计算”保留为模型 planner 候选，不支持据此假设任意模型已经拥有可靠历史追踪能力。[R1]

只有未来改为局部视野、遮挡、动作执行不确定或图像遗漏目标点时，才重新考虑 belief/memory。当前先不扩大问题。

多步未来预测仍保留：从当前状态想象未来 2/4/8 步，是检验规划模型误差的工具，不是独立的历史追踪任务。

## 3. 现有代码的可复用部分与真实缺口

| 模块 | 当前事实 | 本方案用途 |
|---|---|---|
| state/codec.py | fixed/entity 两层状态、七符号编码与 hash | 统一状态接口 |
| state/transition.py | 确定性单步规则，七类事件，1–3 步 action block | 精确转移、预测监督、计划验证 |
| state/validation.py | BFS solver，返回动作路径、最短步数与节点数 | 首版代码求解器与专家数据 |
| engine/env.py | 16×16 tile 的完整 RGB renderer，区分目标上的箱子/玩家 | 图像输入与标签配对 |
| env/core.py、env/session.py | reset/step、快照回放、终止与超时 | 闭环执行器 |
| reward/metrics.py | exact、entity F1、changed-cell F1 | 复用状态评测语义 |
| 现有生成式 Dataset | 给定图片与动作，生成 perception/prediction | 原始多模态模型 baseline |
| Pev frozen features/小型 head | 文本特征与集合策略 | 读出与训练设计参考 |
| env/verl_agent_loop.py | 多轮 VERL adapter 尚未实现 | 不能视为已完成的自主控制训练框架 |

现有 BFS 在 ≤64 cells 时使用 bitboard，更大棋盘回退到 NumPy reference；节点预算用尽返回 solver_unknown，不等于无解。它优化原子动作步数，不是最少推箱数。第一版应直接复用，并独立验证路径，无需让模型重新发明搜索程序。

现有 apply_actions 只接收 1–3 步，长计划验证应新增逐步 replay API，成功即停止，不改变原数据合同。

Pev 的 FrozenQwenBackend 只保留 text backbone；视觉状态模型必须保留图像 processor、视觉塔与融合路径。现有 VERL verified=false 的启动限制继续有效，不为此方案绕过它。

实施主体建议是 Sokoban 项目，保持 Pev 原实验完整。当前只是改写规划文档，不新增训练或求解实现。

## 4. 三个接口与一个执行循环

### 4.1 状态接口

沿用两层语义：

- fixed：wall=0，floor=1，goal=2；
- entity：empty=0，player=1，box=2；
- 展示符号：#、_、O、P、X、+、*，O 为大写字母；
- 坐标：(row, column)，从左上角零起点计数。

StateEstimate 应保留 raw probabilities、decoded board、validity diagnostics、图像/模型版本、state hash。每格置信度只用于诊断和校准后的复核策略，不能把 softmax 高分直接当作整个棋盘正确的证明。

三类接口：

- perceive(rgb, public_geometry) → StateEstimate
- predict(state, actions, horizon) → PredictedStates
- plan(state, goal, budget, dynamics_backend) → PlanResult

PlanResult 包含动作序列、规划时的 initial-state hash、是否已验证、搜索预算与状态：solved / partial / no_plan_within_budget / invalid_input / proven_unsolvable。预算内找不到解不能冒充 proven_unsolvable。

### 4.2 主循环

RGB → 当前状态估计 → planner → 验证候选计划 → 执行首动作 → 新 RGB → 状态核对。

计划本身可以包含整段动作，但首版只执行一个原子动作后检查。若新观测与预期后继一致，可复用未执行的计划；不必每步从零搜索。若不一致，停止执行旧计划，重新识别和规划。

这不是恢复整段历史：只保存当前计划、游标和预期后继即可。禁止将“计划已经执行完”当作成功，实际环境完成标志和评测真值才决定成功。

未来可批量执行经验证的短前缀，但每步观察闭环先作为共同基线。错误推箱可能不可逆，事后重规划并不保证能恢复。

### 4.3 输入边界

真实 StateSnapshot/event/solver labels 由 evaluator 或训练监督端持有，视觉模型只收到 RGB 与公开信息。Oracle-state 是明确标记的实验条件，不能悄悄传给普通 planner。

结构合法不代表状态正确；solver 能解预测棋盘也不证明预测棋盘与图片一致。若识别非法，应返回 invalid/reobserve，不能自动用真状态修好。若识别合法但 planner 失败，应区分感知错误、搜索预算不足和真实无解，而不是为获得一个解修改棋盘。

## 5. 当前棋盘识别：保留 Pev 思想，改为并行 cell readout

### 5.1 推荐候选

完整 RGB 经一次编码，保留一组有空间信息的 tokens；为所有格子创建坐标 queries，经小型 cross-attention decoder 并行输出状态：

RGB → frozen multimodal tokens → coordinate queries → per-cell class logits

七类头输出 [B,H,W,7]，在类别上 softmax；两层头输出两组 [B,H,W,3]。第一轮比较后固定一种，七类头天然避免墙上有玩家之类的非法局部组合，但不保证全局只有一个玩家。

保留 Pev 的每格独立 prompt 分支作为小规模对照，而非默认最终路径。36–64 个重复编码分支与单次编码的成本差异需要实测，batching 不等于共享前缀复用。

明确坐标与 valid-cell mask，不能将 query 列表位置当成空间含义。视觉 patch 不等于棋盘 tile，processor 的 resize/padding/patch 合并必须审计。

先冻结骨干训练小读出模块。若表现不足，按“标注/坐标问题 → feature 层选择 → decoder 容量 → 局部微调”的顺序诊断，再考虑 LoRA 或更大范围 SFT。当前不选硬件，也不启动训练。

### 5.2 必需基线与目标

- 固定 sprite 的模板匹配或小型 CNN；
- 原始多模态模型直接输出状态；
- 冻结骨干 + cell decoder；
- 必要时加入随机初始化空间模型。

目标是整张棋盘正确，尤其是玩家、箱子和被占据目标点。纯逐格准确率可能被墙/地板占比掩盖。保留原始输出与约束后输出两套成绩；约束只能用公开规则与输入信息。

若固定 renderer 的简单 baseline 已非常强，大模型的价值需要靠换皮肤、尺寸变化或更复杂观测证明。不能由使用了 Qwen 推定泛化优于简单视觉模型。

## 6. 棋盘预测：学习什么，为什么保留

### 6.1 明确预测对象

主目标是结构化下一棋盘，不是生成下一帧 RGB。需要展示图片时，将预测棋盘交给 renderer；不引入额外像素生成模型。

定义两种可替换 dynamics backend：

1. ExactDynamics：已有规则代码，确定性计算后继；
2. LearnedDynamics：学习的状态/动作条件模型，输出下一棋盘概率。

两者返回同一状态语义、相同动作顺序。planner 可替换 backend，便于衡量 learned model 的实际误差和成本。

### 6.2 网络与监督

首版 LearnedDynamics 使用小型 action-conditioned CNN 或 cell Transformer，输入当前显式状态，训练环境生成的精确后继。它可以是独立新模型，也可以在后续共享状态编码器；先独立更容易定位问题。

同一状态的四动作反事实可以并行：

output: [B,4,H,W,C]

四个分支都从同一状态开始。先用真状态测纯动力学，再换预测状态测感知误差传播。也可增加 RGB + action → future board 的直接预测对照，但必须与显式 state bottleneck 分开。

训练以逐格 CE 为主，增加变化格子监督和多步展开；报告 no-op、普通移动、推箱、箱子进/出目标、玩家进/出目标。变化 mask 来自标签，只用于 loss。Copy-current 必须作为 baseline。

### 6.3 多步预测属于未来规划诊断

从当前状态预测 1/2/4/8/16 步，用自己的输出递推，测整图 exact、首次错误深度和非法状态率。Teacher-forced 单步与开放环多步分开；完成后的路径停止，不生成成功后的额外动作。

静态地形可以从输入继承，这是明确的归纳偏置；不要把直接复制 fixed 层算成学习到了所有状态内容。

还要收集 search 实际探索的合法状态、死路和非专家动作。只在成功轨迹上预测准确，不代表搜索时不产生“虚假的可解后继”。

### 6.4 对解题的价值边界

规则已知时，精确代码通常是更可靠的转移 backend。学习一个近似模型不是可靠解题的前置要求。本方案保留棋盘预测，是为了检验世界模型能力，而不是人为禁止使用已有规则。

要声称 prediction 帮助 planning，需要：

- planner 确实消费预测后继；
- 与相同 policy/value/search 的无预测或不同 dynamics 对照；
- 报告真实执行成功率，而非只在模型想象中“成功”；
- 显式记录是否使用 ExactDynamics 验证或纠错。

离散模型结合学习 heuristic 与搜索已有直接相关研究，如 DeepCubeAI；它强调多步误差和重复状态识别的问题，值得作为方法参考，但其数据分布与 latent 表示不同，不能直接套用结果。[R2]

## 7. 解题路线 A：状态识别后接代码求解器

### 7.1 推荐作为第一条完整主线

RGB → StateEstimator → 结构化棋盘 → 通用 solver → 动作列表 → executor

规则固定时，不需要每道题生成一份新程序：将识别到的棋盘作为数据传给同一个 solver 就可以。当前仓库已经有 BFS，因此第一阶段重点是状态识别与接口。

### 7.2 搜索如何逐步升级

1. **当前原子动作 BFS**：先用于小图、专家标注、验证最短动作路径。
2. **规则化剪枝和去重**：状态 hash、避免重复状态、可证明静态死锁。
3. **Push-based search**：搜索“哪只箱子向哪推”，由可达性搜索补齐玩家走到推箱位置的路径。
4. **A*/IDA*/weighted search**：按瓶颈再加启发式与成本权衡。

Push-based 状态可在固定箱位下合并玩家同一可达区域，用于最少推箱数搜索；这不保证原子动作最短，必须恢复走路路径并分别报告 moves 和 pushes。若目标是总步数最优，需要保留正确的玩家位置与边代价。

启发式可从忽略其他箱子的反向推箱距离、箱子与目标的最小匹配开始；不可达匹配和非目标角落等可证条件用于剪枝。不能把“箱子已经在目标上”默认当作永久不可移动，因为有些解需要先移开。

Sokoban 的专门搜索研究表明 deadlock 与领域知识对搜索很重要，因此超出 BFS 能力时，优先改搜索结构与启发式，而非默认换成更大 LLM。[R3]

### 7.3 如何理解“让模型写代码解题”

可以做，但应区分：

- **开发阶段生成通用 solver**：模型帮助写一次，测试正确后固定，所有新图输入数据。推荐。
- **运行时调用已有 solve 工具**：模型或控制器传入状态，取回路径。推荐作为总 planner 的工具。
- **每题现场写新程序并运行**：保留为实验对照，测代码生成正确率、运行成功率、搜索代价与真实解题率，不作为默认路径。

若测第三种，固定输入/输出合同和执行资源限制；生成程序返回 actions，由独立 verifier 检查。程序打印“solved”不是成功证据。记录完整代码、调用预算、运行时间和重试次数。不能让生成代码自行改 verifier 或读取隐藏真值。

如果模型生成的是规则 transition 程序，则它是“程序形式的动力学模型”，需要用环境转移进行差分测试；不能与神经 LearnedDynamics 混合命名或计算成绩。

### 7.4 优点、局限与验收

优点：无需训练解题策略；错误来源清晰；能生成专家轨迹与计划证书；最贴近用户“知道当前状态后写代码解决”的需求。

局限：成功主要来自感知与符号搜索，不能声称神经模型自己学会了完整规划；复杂图仍有搜索成本；错误棋盘会让程序正确地解决错误问题。

验收：oracle-state solver 与 predicted-state solver 在同一地图上配对；记录计划在输入状态上验证是否成功、实际环境是否完成、差异出现在哪一步。

## 8. 解题路线 B：单独训练一个解题模型

不必再选一个通用大语言模型。首选基于结构化棋盘的小型空间网络；视觉模型负责观测，解题模型负责状态价值/动作偏好。

### 8.1 三种输出，推荐先做 heuristic

| 输出 | 如何使用 | 推荐程度 |
|---|---|---|
| policy π(a|s) | 直接选四动作，或给搜索排序 | 必需对照 |
| heuristic h(s) / value V(s) | 引导精确规则搜索 | 优先，能保留已知转移的正确性 |
| 整段动作计划 | 自回归生成完整解法 | 可做 baseline，长序列错误需独立回放 |

重点是“**学搜索方向，比重新近似已知物理规则更可能直接帮助解题**”这一待验证判断。DeepCubeA 的作者实现提供 learned cost-to-go 与搜索结合的 Sokoban 路径，支持这条路线的可行性，但不能由此预测本项目收益。[R4]

### 8.2 数据与训练顺序

- 用精确 solver 的成功路径生成 BC targets 和 cost-to-go 标签；
- 区分最短动作距离、最短推箱距离、普通可行解上界；求解超时是 unknown，不填成负例或无穷距离；
- 对等价最优动作尽量使用 optimal-action set，避免惩罚不同但同样正确的动作；
- 学生访问状态上补专家标签，逐步缓解分布偏移；
- 有稳定 BC/heuristic 基线后，才比较 RL 是否进一步改善。

learned heuristic 未必 admissible。若用于 A*/weighted search，要声明不保证最短解；需要保留最优性时，可先只作 admissible heuristic 的 tie-breaker，或明确维护有效下界与终止条件。

相同 solver 与节点预算下比较 handcrafted heuristic、learned heuristic、policy ordering；同时比较实际运行时间。神经评分减少节点数，不一定降低墙钟时间。

### 8.3 控制训练是可选扩展

直接 policy 可做 BC → 数据聚合 → PPO。四原子动作始终保留，no-op 是环境结果。真实成功奖励与轻微 step cost 为起点；不靠来回进出目标累计正奖励。

PPO 使用真实环境 rollout，持久 session 跨 update 延续；rollout 边界 bootstrap，任务 deadline 失败不 bootstrap。remaining_steps 作为公开条件。不要复用 Pev 每轮重置全部环境的短任务采样方式作为长解题默认实现。

## 9. 解题路线 C：总模型作为 planner

“一个总模型”有两种不同含义，必须分开。

### 9.1 C1：模型编排工具，推荐的高级系统路线

总模型读取当前 StateEstimate 和任务目标，可：

- 请求一次重新识别或局部复核；
- 调用 ExactDynamics 或 LearnedDynamics；
- 调用 solve/search，设置预算或候选子目标；
- 请求 verify_plan，检查失败位置；
- 选择执行已核对的动作前缀；
- 使用新观测触发重规划。

~~~mermaid
flowchart TD
    I[当前 RGB] --> P[状态提取模型]
    P --> S[当前棋盘与置信信息]
    S --> C[总 planner / 固定控制器]
    C --> X[精确 solver]
    C --> W[学习的棋盘预测]
    C --> H[独立 policy / heuristic]
    X --> V[计划与转移验证]
    H --> V
    W --> C
    V --> E[执行首动作]
    E --> I
~~~

工具返回结构化结果，总模型的自述不能覆盖 verifier 的结果。ExactDynamics/verifier 只能检查相对于输入棋盘与已知规则的正确性，无法证明图像识别正确。

LLM+P 展示了将问题结构化后交给经典 planner 的组合方式；LLM-Modulo 讨论候选生成与外部验证的协作。这些是架构依据，不是它们在本项目 Sokoban 上已有成绩，也不意味着今日所有 LLM 都无法规划。[R5][R6]

首个 orchestration baseline 是固定控制器，不加 LLM。只有复杂故障恢复、工具选择或预算分配显示收益，才证明总模型有必要。不能只因链路加了 LLM 就视为能力提高。

### 9.2 子目标如何使用

模型可建议“先把这只箱子移开以打开通道”等子目标，但只作为搜索偏好；如果不可行，必须能放弃。

Sokoban 中局部目标可能冲突。将每个箱子贪心放到最近目标并锁定，可能导致死锁。不能把自然语言子目标未经可达性/解题验证就作为强制操作。

为避免模型循环讨论，固定 planner calls、search nodes、重试次数与总时间；所有尝试计入成本。节点预算耗尽与状态错误要有不同返回码。

### 9.3 C2：共享骨干的统一神经模型，后续研究路线

同一骨干/状态编码上接：

- perception head；
- action-conditioned next-state head；
- policy/value head。

共享权重不意味着只需一次 forward 完成任意深度搜索。向前展开仍需要多次 dynamics 计算；policy head 也可能绕过显式 state head。

建议比较：
1. policy 只能读取预测状态；
2. policy 可读取 latent；
3. latent policy 加/不加状态与动力学辅助 loss；
4. 分离模型与共享模型，在匹配数据/计算预算下对照。

先独立训好模块，再考虑多任务联合优化；同时监测动作 loss 是否损害视觉状态准确率。内部 recurrent thinking 可增加，但不重新引入历史追踪任务。

## 10. 解题路线 D：以学习到的棋盘预测驱动搜索

这条路线用于明确回答“预测棋盘能不能帮助解题”，与可靠代码 solver 并列研究，不挡住工程交付。

RGB → 预测当前棋盘 → LearnedDynamics 扩展搜索树 → heuristic/value 评价 → 首动作 → 真实新图。

先从深度 2/4/8 的固定预算 lookahead 开始，再考虑 beam、best-first 或 MCTS。搜索算法先固定，改变 dynamics backend；不要同时换 perception、heuristic、搜索宽度后声称收益来自预测。

三种规则访问条件必须分别报告：

| 条件 | 搜索后继 | 计划验证与反馈 | 可以得出的结论 |
|---|---|---|---|
| Exact | 精确规则 | 精确规则 | 感知/heuristic/solver 系统性能 |
| Learned-only | 只用 LearnedDynamics | exact 仅作不反馈给 planner 的离线评测；真实执行后看观测 | 学习模型独立支持规划的能力 |
| Hybrid | 学习预测 | 允许精确 verifier 返回错误并修复 | 有规则工具辅助的混合系统性能 |

相同 learned plan 被 exact verifier 拒绝并触发重试后，就不属于 learned-only。所有工具调用与验证成本都应记录。

由于已有真实规则，learned dynamics 不太可能在精确性上胜过它。此分支的价值在于验证模型的转移能力、规划利用能力，以及后续未知规则场景的潜力；本轮不把跨规则泛化作为必须新增的任务。

## 11. 研究结论如何对应架构选择

| 需求 | 首选路线 | 暂时不需要 |
|---|---|---|
| 尽快稳定解题 | 状态提取 + 代码 solver | 新大模型、PPO、历史追踪 |
| 提升复杂图的搜索效率 | 独立 heuristic/policy + 精确搜索 | 每题重写 solver |
| 研究棋盘预测是否有用 | learned dynamics + 受控搜索消融 | 用真规则偷偷纠错 |
| 需要统一调度、恢复和解释 | 总模型 + 结构化工具 + verifier | 要求 LLM 手算所有转移 |
| 研究单模型内部规划 | 共享多头 + direct policy/recurrent computation | 第一阶段就大规模端到端 RL |

建议优先级：A → B 的 heuristic 分支 → C1。棋盘预测能力与 A 同步建设，D 作为主研究对照；C2 和直接策略 PPO 由前面结果决定是否值得投入。

## 12. 数据与课程

### 12.1 三套监督，共享物理状态

| 数据 | 输入 | 标签 |
|---|---|---|
| Perception | RGB | 当前 fixed/entity |
| Prediction | 当前状态 + 动作，或 RGB + 动作的单独对照 | 下一状态、事件、终止 |
| Planning | 当前状态与目标 | 专家动作集合、计划、代价、求解状态 |

每个非终止状态优先补齐四动作转移，而不只收集专家动作。状态预测数据允许合法但不可解的状态；规划专家标签只对已证实可解且成功求解的状态建立。solver_unknown 单列。

轨迹用来提供连续的未来预测样本与规划监督，不要求模型消费全部动作历史。初始样本来源用专家路径、随机探索、扰动路径；后期加入实际搜索节点和 learner 状态。

### 12.2 覆盖与划分

覆盖全部七种格子组合和七类 event。单箱到目标即成功，若要采集非终止“箱子离开目标”，使用多箱状态，遵守完成后停止的协议。

先 6×6、单箱、短解，再 7×7/8×8、双箱、绕路和目标暂时腾空。难度按已验证最短步数 1–8/9–16/17–32/33–64 分桶；reverse generation depth 不是最短解长度。

按 map group 划分 train/dev/test；同图派生状态、旋转、镜像、图片皮肤、未来分支全部同 split。新 renderer、新尺寸、新箱数分别作为 OOD 条件，动作方向随几何变换同步变化。

可以从约 50–100 图的合同 smoke 开始，首轮 pilot 候选为约 2,000 训练图、每图最多 20 个去重状态，dev/test 各 200 图。数字仅作规划，不启动生成。正式规模由覆盖、搜索成本与 learning curve 决定。

### 12.3 成本和来源

由环境生成的 state/transition 是准确监督，不需要 35B teacher 再猜一次。solver 提供专家轨迹，其调用成本单列；生成式 LLM baseline 的文本、工具和执行成本单列。

新增 manifest 版本，不修改已有 1–3 步数据合同。保留 state/image/code/model/solver hashes，模型请求与监督字段分开序列化。

## 13. 最小实验矩阵与指标

核心对照按阶段补齐，不要求首批同时运行：

| 组别 | 感知输入 | Planner | 用途 |
|---|---|---|---|
| A0 | 真实棋盘 | 当前精确 BFS | 求解器参考 |
| A1 | 图像估计棋盘 | 与 A0 相同 | 感知带来的解题损失 |
| B0 | 真实棋盘 | learned heuristic + 精确搜索 | 学习能否帮助搜索 |
| B1 | 图像估计棋盘 | 与 B0 相同 | 完整神经感知 + 学习搜索方向 |
| D0 | 真实棋盘 | learned dynamics + 同预算搜索 | 纯转移误差影响 |
| D1 | 图像估计棋盘 | 与 D0 相同 | world model 闭环 |

后续按问题追加：direct policy、原始 LLM 直接计划、每题生成代码、固定工具控制器、LLM 总 planner、统一模型。所有组先在真状态上测 planning，再用固定视觉模型测端到端，避免混淆感知和规划。

主指标：

- 感知：whole-board exact、玩家定位、箱子集合 exact、目标识别、非法状态率；
- 预测：一步及 2/4/8/16 步 exact、changed-cell F1、no-op exact、事件分桶、首次模型错误；
- 规划：预算内真实 solve rate、计划回放通过率、解步数/推箱数、首个错误动作、可证死锁/超时；
- 系统：首次解耗时、节点数、backbone/LLM calls、tokens、verifier 次数、重试次数、GPU/CPU 总时间；
- 总模型：相对固定控制器的增益，以及是否只是使用了更多搜索/重试预算；
- 代码生成：程序可运行率与实际解题率分开，生成/修复时间计入总预算。

逐格高准确率不能代替整图 exact。一次无效转移可能导致虚假的目标状态，因此模型内预测成功和真实成功分开。

正式学习对照至少三个训练 seed，相同地图配对；区间按地图聚合，不把格子或同图状态当作独立样本。同时提供固定节点预算和固定时间预算结果；LLM 工具调用不能被视为免费算力。

## 14. 推进阶段与验收

| 阶段 | 交付 | 推进条件 |
|---|---|---|
| S0 接口与基准 | 三接口、输入隔离、现有 solver 路径回放、固定 split | 真值回放和终止语义通过，unknown 无误标 |
| S1 当前棋盘 | 并行 cell decoder 与视觉 baseline | easy 固定 renderer 的 raw whole-board exact 候选目标≥95%；同时看闭环损失 |
| S2 第一版自主系统 | 感知 + 精确 solver + 单步观测执行 | 真/预测状态同图配对，动作/观测/计划 hash 对齐，错误可定位 |
| S3 棋盘预测 | 一步、四动作批量、多步未来预测 | 候选目标：easy 一步 exact≥99%，各事件单列；多步误差可测 |
| S4 学习帮助搜索 | learned heuristic 与 learned dynamics 分开接入 | 同图同预算下比较收益与成本，确定真正贡献 |
| S5 总 planner 对照 | 固定控制器、LLM 工具编排、每题代码生成 | 同工具与重试预算比较，验证环节不能偷换 |
| S6 可选模型合并 | 统一多头/直接 policy/RL | 前面基线齐全，联合优化不破坏感知与预测，泛化可复现 |

这些数字是开发门槛候选，不是已有成绩，也不是充分条件。95% 单图识别在长 episode 上仍可能不可靠，应结合真实闭环与误识别恢复评估。正式阈值在 dev 上确定，test 不用于反复调参。

S2 不必等待 S3：这是本次修订的重要变化。用户可以先得到可靠解题系统，再决定学习模型 planner 值不值得做。历史追踪不再是任何阶段的依赖。

第一批实施范围建议只到 S0/S1，并为 S2 留接口；目前仍停留在规划。

## 15. 代码模块与后续实现边界

建议新增而不破坏已有生成式训练路线：

- perception/：multimodal encoder、cell queries、decoder、state estimate；
- dynamics/：exact adapter、learned model、batched counterfactual、future rollout；
- planning/：solver wrapper、plan verifier、heuristic search、可选 LLM orchestrator；
- control/：执行器、计划游标、状态核对、replan；
- evaluation/：固定状态预测、规划矩阵、闭环配对、错误定位；
- datasets/：三任务视图与 split 绑定；实际路径与现有 package 结构统一。

关键测试：坐标与图像对齐、query permutation/padding、目标点覆盖、四动作分支、无隐藏 state 输入、预算/终止、计划从正确 initial hash 回放、wrong-perception case、solver_unknown、macro 与原子路径一致性、learned-only 无 exact 反馈。

长计划 verifier 逐步调用现有 apply_action，并在完成时停止，不绕开 apply_actions 的 1–3 步旧合同。当前 session 可直接复用，PPO collector 则需专门实现持久 episode。

前三项能力的小型模型可以先用 PyTorch；只有确定需要分布式生成式训练时才扩展 VERL。冻结特征可缓存，解冻后缓存失效；记录模型与 processor 版本、读出层和 kernel。

现有骨干选择保持可替换，不假定所有模块必须使用 Qwen3.5-4B，也不假定“总模型”必须比视觉模型更大。

## 16. 研究依据及可迁移范围

以下只作为方法依据，未将论文的成功率当作本项目预期值：

- **[R1] 内部计算与历史是不同问题。** [An Investigation of Model-Free Planning](https://arxiv.org/abs/1901.03559) 研究 recurrent 网络及额外 thinking time；可借鉴其规划计算对照，不要求训练历史重建。
- **[R2] 学习离散模型后做搜索。** [Learning Discrete World Models for Heuristic Search](https://rlj.cs.umass.edu/2024/papers/Paper225.html) 与 [作者 DeepCubeAI 代码](https://github.com/misaghsoltani/DeepCubeAI) 结合离散 world model、学习 heuristic 和搜索，涉及 Sokoban。其重点提示本项目关注多步误差和状态去重。
- **[R3] Sokoban 搜索需考虑死锁与领域结构。** [Junghanns & Schaeffer, Sokoban: Enhancing general single-agent search methods using domain knowledge](https://www.sciencedirect.com/science/article/pii/S0004370201001096)。不能把大图求解简单等同于无剪枝 BFS。
- **[R4] 学习启发式而保留精确规则。** [DeepCubeA 作者实现](https://github.com/forestagostinelli/DeepCubeA) 包含 Sokoban 的 learned cost-to-go 与搜索，可用于独立解题模型路线参考。
- **[R5] 结构化问题后交给程序 planner。** [LLM+P](https://arxiv.org/abs/2304.11477) 用形式化问题与经典 planner 组合；本项目已有固定规则与状态 schema，第一版不必额外绕一层 PDDL。
- **[R6] 候选生成与外部验证分工。** [LLM-Modulo](https://arxiv.org/abs/2402.01817) 是组合规划的立场与框架参考，不将标题中的绝对判断当成对所有当前模型的实验证明。
- **[R7] 模型辅助决策的另一路线。** [Imagination-Augmented Agents](https://arxiv.org/abs/1707.06203) 可参考如何利用预测 rollout；第一版优先显式棋盘与可检查搜索。
- 本地证据：state/transition.py、state/validation.py、engine/env.py、env/session.py、reward/metrics.py、Pev 的 report_new.md。已有计划中关于数据版本、VERL runtime 验证和输入边界的限制保持有效。

## 17. 本次修订后的最终建议

先把“看清棋盘”做成稳定的状态接口，再直接复用代码求解器实现自主执行。同时训练并评估“行动后棋盘怎样”的模型，通过替换转移 backend 检验它对搜索的价值。

第二个新模型优先学习 heuristic/value 或动作偏好；总模型优先做有工具和验证的编排。每题重新生成代码、完全统一端到端 planner 都保留为有预算边界的对照。

取消历史追踪作为独立任务。保留未来预测和搜索计算状态，但不让它们成为新的无关目标。
