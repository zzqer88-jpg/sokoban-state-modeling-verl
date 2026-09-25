# Sokoban 当前状态识别：视觉 SFT 与 Jev 接入规划

当前实施范围更新：在本项目新增 `src/sokoban_state_modeling/jev/`，仿照 Jev 的 models/prompts/backend/scoring/service/config/cli 命名，并补充 dataset/training/evaluation。第一阶段直接使用 Transformers/PyTorch，暂不引入 SGLang；本文有关共享前缀预热和服务端改造的内容属于后续可选部署。目录职责、阶段交付和验收以 [JEV_VISUAL_MODEL_DESIGN.md](JEV_VISUAL_MODEL_DESIGN.md) 为准。正式训练尚未启动。

模型改造细节见 [JEV_VISUAL_MODEL_DESIGN.md](JEV_VISUAL_MODEL_DESIGN.md)，按 Pev 原生读出与 ChoicePolicy 两条路径分别核对。当前沿用远程 `data/sftcoldstartv11/sft/` 的 SFT 数据，原始规模目标为 train 12,000 / validation 1,200 / test 3,000；转换去重后以 audit.json 为准，替代下文早期建议规模。

日期：2026-09-25 更新。概率训练与本地推理代码已实现，未启动正式训练。此文件细化当前生效范围；动力学与解题不在本阶段内。

实现更新：已新增数据转换器 `scripts/export_jev_sft.py` 和 `jev/` 的本地概率训练/评估模块；SGLang HTTP 视觉服务不在当前范围。使用命令见 [JEV_USAGE.md](JEV_USAGE.md)。当前工作区没有现成 SFT JSONL，因此尚未批量转换真实数据。

### 从已有 SFT 数据导出

```bash
PYTHONPATH=src python scripts/export_jev_sft.py \
  --input /path/to/train.jsonl /path/to/val.jsonl /path/to/test.jsonl \
  --output outputs/jev_perception
```

输入兼容现有 thinking SFT JSONL 和含 `current_state.grid`、`split` 的原始 manifest；只使用当前状态，不解析旧思考或预测答案。输出目录必须不存在。应同时传入所有 split，以检查地形地图及其旋转/镜像变体的跨 split 泄漏；该检查不等价于所有生成来源家族的完全审计。相同当前状态去重，保留第一条来源位置，并在审计报告记录去重数量。

输出包含 `records.jsonl`、`images/*.png` 和 `audit.json`。每条记录的 `input` 包含图片引用、英文 state、英文 questions 与按 64 个问题划分的 request_groups；`target.answers` 保存各坐标的七类 one-hot 概率，`class_order` 固定顺序。provenance 保存来源行号、状态/地图哈希、渲染版本与图片哈希。加载图片时相对导出目录解析路径。

这是供训练与未来多模态 Jev 适配器使用的数据契约，不是能直接 POST 给未改造 Jev 的 HTTP 请求。适配器需逐组选择 questions，把图片引用加载为真实多模态载荷；target/provenance 不能进入模型输入。原版普通 token SFT trainer 不会自动计算这种 KL 损失，本项目现已提供独立 PyTorch 概率训练器。数据导出测试已覆盖七类映射、图片渲染、重复状态、镜像地图泄漏、哈希损坏和 81 格问题分组。

## 服务端源码核对补充（优先于下文待核实描述）

已通过 GitHub 读取公开 `ekzhang/openjev-sglang` 的固定提交 `f3e1678168b2e9a298bb47639444b428b661c4b2`，其包名和接口与 Pev 调用一致；尚未确认原实验部署使用的确切提交。这是静态源码审计，未运行 GPU 服务。保留七类方案。

- `prompts.py`：生成公共 prefix_ids，每个问题生成独立 Branch，input_ids 为 prefix + suffix；不同问题不包含彼此问题或答案。question_id 只用于返回映射，不进入提示，因此坐标必须写进 instructions，不能只写在 cell_2_3 这样的键名里。
- `service.py`：先 await 公共前缀的一次 generate，预热后才 create_task 各分支并 gather。一个 API 请求的 N 个问题实际对应 N+1 次内部 generate 调用，不是一次模型 forward。各分支由 SGLang 调度，不能保证全部同处一个 GPU batch。
- `backend.py`：每次 max_new_tokens=1，使用 token_ids_logprob 一次读取该问题所有类别标签的概率。`scoring.py` 对这些标签重新归一化并取最大值；实际采样到的 token 不用于决定返回类别。每格七类不扩成七个分支。
- `runtime.py`：启动参数要求 radix cache，使用 mamba-radix-cache-strategy extra_buffer。README 明确这是机会式复用，非锁定的每请求缓存；不能保证前缀永不重算。每个分支仍提交完整 input_ids，公共前缀通过后端缓存命中复用。
- `models.py`：每请求最多 64 个问题，每个 choice 最多 64 类，是两个不同限制。5×5 的 25 个问题在数量限制内；9×9 的 81 个问题需要分请求或修改限制，还需满足总输入预算。默认全局后端并发数为 64，并非每请求独占 64。
- 当前版本明确不支持视觉：`state_messages` 对非 text 的聊天内容抛出 `Jev state supports text content only`；对应测试明确验证 image_url 被拒绝。编译器使用 tokenizer；backend 只传 input_ids 和评分参数，没有图片载荷。把图片写成普通 JSON 只会变成文本，不能获得视觉能力。
- 视觉接入必须修改请求、图文编译/processor、后端多模态载荷及缓存处理。图片内容需参与缓存标识，warmup 与分支必须保持多模态位置/输入一致；分别验证图像编码缓存和语言前缀缓存，不能以文本 radix 命中证明图片只编码一次。

因此部署目标是“一张图片作为共享 state + H×W 个 choice 问题”，沿用 Jev 先预热再并发分支的结构；当前版本还不能直接执行图像任务。独立分支 SFT 与该条件分解相符，但服务端推理缓存不直接等价于可反向传播的训练缓存。

固定源码：[服务流程](https://github.com/ekzhang/openjev-sglang/blob/f3e1678168b2e9a298bb47639444b428b661c4b2/src/openjev/service.py)、[问题编译与视觉拒绝](https://github.com/ekzhang/openjev-sglang/blob/f3e1678168b2e9a298bb47639444b428b661c4b2/src/openjev/prompts.py)、[后端评分请求](https://github.com/ekzhang/openjev-sglang/blob/f3e1678168b2e9a298bb47639444b428b661c4b2/src/openjev/backend.py)、[缓存说明](https://github.com/ekzhang/openjev-sglang/blob/f3e1678168b2e9a298bb47639444b428b661c4b2/README.md)。

## 1. 明确模型任务

输入完整、无遮挡、轴对齐的棋盘 RGB、行列数以及查询坐标。输出该格七类状态概率；对全部坐标查询后组装当前状态。棋盘尺寸由环境提供，第一版不要求模型从任意截图定位棋盘或推断行列数。坐标零起点，左上角为 (0,0)，行向下、列向右。

七类沿用现有 codec 语义：wall (#)、floor (_)、goal (O)、box (X)、box_on_goal (*)、player (P)、player_on_goal (+)。符号是内部编码，分类答案优先使用 Jev 编译器已支持的标签 token。视觉输入必须能区分目标被玩家或箱子占据的情况；若某个渲染主题完全隐藏底层目标，则不能从单张图片唯一恢复状态，应修正渲染或排除该主题，不要求模型猜测。

## 2. 数据生成和划分

数据单元是一张完整棋盘及其完整标签矩阵，不是独立收集的格子小图。由同一个 StateSnapshot 同时生成 RGB 和标签，保留 state_hash、map_hash、地图来源、轨迹来源、renderer 版本、图像 hash 和随机种子，以便复现。

先按地图家族划分 train/validation/test，再采样局面。同一地图的旋转、镜像版本及其轨迹必须在同一集合。map_hash 的地形定义不足以自动识别所有变体，应另外存放来源 family_id 或变换规范化哈希。不能先展开坐标问题再随机划分。相同局面对应不同历史动作或下一动作时，只保留一个基础状态；增强视图可多个，但要能追溯同一基础样本。

局面来源混合：初始状态、合法随机行走/推箱产生的状态、已有解法轨迹中的中间与完成状态，以及针对稀有类别构造并验证的状态。识别任务可以包含死锁局面，不能只收录可解或容易解的状态。禁止随机逐格贴标签产生互相冲突的棋盘。直接构造的状态需标注来源并检查玩家数、实体与墙互斥及项目规则要求的箱子/目标数等不变量。

优先覆盖：箱子/玩家在目标上；多个相邻箱子；贴墙与角落；同一地图中实体移动而背景不变；不同目标位置；非方形与不同尺寸。针对关系学习保留合法的成对局面：背景相同、玩家或箱子改变位置。它们仍作为独立图片接受逐格监督，不输入前后动作，也不引入预测损失。

建议初始规模（实验建议，非现有数据或已验证最佳值）：

| 阶段 | 训练 | 验证 | 测试 | 用途 |
|---|---:|---:|---:|---|
| 管线检查 | 100–200 张 | 少量独立地图 | 暂不做正式结论 | 检查图像、坐标与标签对齐，未来训练时验证能否拟合 |
| 第一轮 | 10,000 张 | 1,000 张 | 2,000 张 | 检验逐格与整盘识别 |
| 按错误扩充 | 30,000–50,000 张 | 固定验证集 | 锁定测试集 | 定向增加稀有类别和失败场景 |

第一轮训练集争取至少 1,000 个不同地图家族，每个家族限制采样量，避免少数地图贡献绝大多数样本。实际可用地图不足时如实记录，不将轨迹数量当地图多样性。所有数量按完整棋盘计；6×6 的一万张棋盘对应 36 万个潜在坐标监督，不是 36 万张独立图片。

第一版使用标准 renderer；先验证 processor 实际 resize 后仍能看清每格及覆盖目标的细节，再固定像素预算。不要只规定一个任意分辨率。后续添加语义保持的缩放、边距或轻微颜色变化；旋转/镜像必须同步变换标签和尺寸，且归入原地图家族。禁止随意裁掉棋盘。新主题、新尺寸分别作为独立泛化测试，不与同分布指标混报。

## 3. 数据契约与训练展开

建议基础记录（项目内部格式，不宣称是 Jev 现有 API）：

```json
{
  "schema_version": "sokoban-perception-sft-v1",
  "sample_id": "board_000001",
  "family_id": "family_0001",
  "map_hash": "...",
  "state_hash": "...",
  "image_ref": "images/board_000001.png",
  "height": 6,
  "width": 6,
  "coordinate_system": "zero_based_row_col",
  "labels": [[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 4, 1, 1, 0], [0, 1, 1, 5, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]],
  "label_vocabulary": ["wall", "floor", "goal", "box", "box_on_goal", "player", "player_on_goal"]
}
```

也可以按现有方式保存状态并惰性渲染；state 字段只供数据加载器使用，绝不拼进模型提示。图像路径/文件名不得暴露答案，也不作为提示文本传给模型。独立对渲染图进行抽样可视化审查。

训练时展开为完整图片 + 尺寸/坐标约定 + 单个坐标问题，目标为七类概率分布。例如查询 (2,2) 的真值是 box_on_goal，则该类目标概率为 1，其余为 0。提示包含七类语义表，但不包含其他格子真值、当前字符棋盘、动作、未来棋盘或思考链。答案的实际 token 字符串由 Jev 编译器决定，不能未经验证假定 A–G 在其答案边界上都是单 token。

以棋盘为采样单位。初版遍历全部坐标，使用小批次/梯度累积控制资源；若成本太高，再采用坐标子采样，混合均匀坐标和稀有类坐标，保留采样概率并在需要时做逆概率权重校正。优先在棋盘层补充稀有状态，避免只查询箱子、玩家而导致空间与类别先验偏移。验证与测试一律查询所有格子。

### 3.1 明确的 5×5 输入输出示例

以下字符棋盘仅展示标注真值，模型实际看到的是其 RGB 渲染图：

```text
#####
#_O_#
#_X_#
#P__#
#####
```

一张图片保存一次，附完整 5×5 标签矩阵。训练加载时展开为 25 个独立坐标监督。单个分支的语义输入如下；实际聊天模板和标签必须由与推理一致的编译器生成，不手工替换其特殊 token。

语言约定：模型可见的说明、坐标问题、类别描述全部使用英文，训练与推理一致；类别键使用英文，监督目标为按类别键对应的概率分布；字母标签只用于模型内部读取各候选分数。本文中文只用于方案说明。下例的图片占位说明不是实际输入文本，真实图片通过多模态载荷传递。

```text
[Full-board RGB image supplied through the multimodal input]
The board has 5 rows and 5 columns.
Coordinates are zero-based (row, column), with (0, 0) at the top-left.
Row indices increase downward. Column indices increase to the right.

Question: What is the current state of the cell at (row=2, column=2)?
Options:
A: A wall (wall).
B: Empty floor with no goal, box, or player (floor).
C: An unoccupied goal square (goal).
D: A box on a non-goal floor square (box).
E: A box on a goal square (box_on_goal).
F: The player on a non-goal floor square (player).
G: The player on a goal square (player_on_goal).
Answer:
```

对应监督输出（训练器接收的结构化数值，不是让模型生成 JSON 文本）：

```json
{
  "cell_2_2": {
    "type": "choice",
    "probabilities": {
      "wall": 0.0,
      "floor": 0.0,
      "goal": 0.0,
      "box": 1.0,
      "box_on_goal": 0.0,
      "player": 0.0,
      "player_on_goal": 0.0
    }
  }
}
```

以上为项目训练目标格式，沿用 Pev 的 choice 概率语义，不声称是新增的现有 Jev 请求字段。推理响应另包含 choice=argmax(probabilities)。该样本语义真值为 box；不再采用监督生成单个 D 的全词表 token SFT。A–G 仅作内部候选 token 映射，从七个候选 logits 得到预测概率。输入不含任何目标概率。

完整训练记录额外保存 target_probabilities，按坐标组织全部 25 个上述目标；加载器转换为 [H,W,7]，批处理为 [B,H,W,7] 并对变长棋盘提供有效格子 mask。labels 可作为精确真值和评估字段保留；允许从 labels 无损生成 one-hot 概率，避免冗余字段不一致。类别顺序必须固定，并根据键名对齐，不能依赖不受控的字典顺序。

训练可以批量处理坐标分支，也可以拆成小批次。推理目标是一个多模态 state 加 25 个 choice 问题；坐标明确写进每个问题的 instructions，不能只写在 question_id。沿用已审计的服务结构时，先做一次公共前缀预热，再并发发出 25 个分支；共 26 次内部 generate 调用，每个分支读取七类分数，而不是 175 个分支。视觉载荷与缓存仍需改造验证。

每格的类别与概率由 Jev 返回；程序按坐标汇总，形成下列结果，而非要求模型逐字生成此 JSON：

```json
{
  "height": 5,
  "width": 5,
  "grid": ["#####", "#_O_#", "#_X_#", "#P__#", "#####"]
}
```

同时保存 [5,5,7] 的概率数组以供评估。字符网格仅是状态输出编码，不是输入的字符解析任务。七类方案已确定；地形/实体两层输出暂不采用。

## 4. 模型与并行语义

优先复用完整 Qwen3.5-4B 多模态 checkpoint、配套 processor 和原 LM head。模型版本、processor 配置、聊天模板、Jev 编译器及标签映射一并固定。官方 Transformers 文档要求图文输入使用完整 Qwen3_5ForConditionalGeneration 配合 processor；Pev 当前只抽取 language_model 的实现不能直接复用为视觉后端。

逻辑结构：图片 + 共享说明 → 完整多模态模型；每个坐标问题作为独立条件分支，答案位置的 LM logits → 取七个标签 logits → 格内 softmax → [H,W,7]。每个分支都看全图，但分支之间不读取对方答案。模型参数共享，不为每个格子训练一个模型，也不为七类各跑一次模型。

正确性优先的第一版：普通独立分支批处理，允许重复图片计算。训练与服务均逐分支得到相同条件概率。之后才优化图像前缀共享。Qwen3.5 包含线性/全注意力混合层，不能假设只复制普通 KV cache 或加一个 attention mask 就能支持任意分支；缓存状态、位置编码和梯度语义都需要实现核对与数值对照。推理缓存不能直接当成训练共享方案，尤其不能意外 detach 应被训练的前缀表示。

Jev 是问题编译/受限评分和服务接口，不是另一个需要教师标签的模型。本任务引擎已有准确监督，不需要通过 Jev 蒸馏教师。公开服务源码已审计，详见本文开头：当前只支持文本，最多 64 个问题，标签动态验证，先预热再并发分支；待完成的是视觉扩展及其运行验证。

## 5. SFT 目标与参数策略

主方案改为逐格候选概率监督，参考 Pev 的 `OpenJevTeacher._raw`、`ChoicePolicy.forward`、`teacher_kl` 与 `distill`。Pev 教师返回候选概率，学生构造 Categorical 分布，使用 KL(target || prediction)，不是监督 JSON 字符或单个答案 token。

本任务每格预测 p=softmax(z)，z 是七类分数；目标 q 是七类概率。损失为每格 sum(q * (log(q) - log(p)))，q=0 的项按 0 处理；先在每盘有效格子内平均，再跨棋盘平均。也可使用数值稳定的软目标交叉熵 -sum(q*log(p))，对固定 q 与 KL 的梯度相同。只在七类间归一化，不跨格子归一化，也不用全词表交叉熵。

第一版采用引擎精确标签的 one-hot 概率：正确类 1，其余 0。这是有效的概率分布监督；不编造 0.9/0.1 等软标签。若未来引入经过验证的视觉教师，可保存其完整软概率及来源，另做蒸馏实验，不用教师不确定性覆盖已知引擎真值。每格目标应检查键集合、有限性、非负性与总和为 1。

架构边界：本方案复用 Pev 的候选概率接口和 KL 目标；当前模型仍为完整视觉语言模型，取 LM head 的七个标签分数并在候选内 softmax，训练可用 LoRA。Pev 原学生则是冻结文本特征 + 独立 ChoicePolicy 小模块，且按候选分别编码；不能声称二者结构或成本完全一致。若后续要原样迁移 Pev 学生架构，需另建视觉特征后端和逐格候选编码，并重新核算分支数量。当前不需要 critic、PPO 或动作头。

当前用户已选定服务器 full SFT：视觉编码器、连接模块、语言模型与 LM head 全部训练。下面 LoRA 相关选择仅是可选对照，不是默认方案。这样视觉信息仍参与全部前向计算。适配器的具体模块名需要检查实际 checkpoint；LoRA rank 可从 16/32 两档小规模验证，不视为已选定最优参数。如果目标覆盖细节丢失，先排查渲染和 processor；确认是表示瓶颈后再解冻部分视觉层。全参数微调作为后续对照，不因旧 launcher 默认全参数就直接沿用。

未来训练时先在 100–200 张上验证拟合与坐标对齐，再用独立验证地图比较有限配置。按实际梯度更新统计各可训练模块，防止视觉连接器或 LoRA 意外未参与训练。使用验证集整盘准确率选 checkpoint，测试集仅用于最终报告。

## 6. 关系能力与评估

当前结构能利用全图上下文，但没有跨答案分支的显式联合约束。通过未见地图、非方形棋盘、同地图实体移动、旋转映射和坐标顺序随机化评估空间定位，不能只靠训练损失宣称学会关系。增加去掉图片/打乱图片与样本配对的对照，检查是否只记住墙边界和类别先验；这些对照要单独统计关键实体类别，避免背景准确率掩盖问题。

主指标为整盘完全正确率；同时报告逐格准确率、七类 precision/recall/F1、混淆矩阵、玩家/箱子/目标集合完全正确率和结构合法率。箱子/玩家在目标上的格子要同时计入对应目标集合。按尺寸、主题和局面来源分组，报告不同置信度阈值下的覆盖率与准确率。未经修复的原始输出是主结果，后处理另列。

如必须设工程门槛，可暂以同分布独立测试整盘准确率 ≥99%、七类召回率分别 ≥99% 为目标，但它们不是已有结果，也不保证对新尺寸/主题达标。罕见类别必须报告样本量，测试图像相关性按地图家族考虑。整盘阈值的选择最终取决于后续使用容许的错误率。

若出现持续的全局不一致，再单独比较联合 cell decoder：所有坐标 query 对图像特征做 cross-attention，query 间做 self-attention，然后共享七分类头。它可并行，但属于新增结构，不再是原样 Jev LM-head SFT。本阶段不预先实现它。

## 7. 实施清单与复用边界

1. 审计 Jev 服务源码与实际模型/processor 版本；确认单分支图文输入和标签 token，然后验证多坐标批量结果与逐个调用一致。
2. 增加 perception-only 数据 schema、导出器及加载器；复用 state codec 和 renderer，不修改旧 thinking/prediction 数据的含义。
3. 增加单坐标训练编译器；与 Jev 服务编译器共享配置或直接复用，实现 prompt/token 对照。
4. 增加独立概率监督配置、参数冻结策略、候选 logit 读取与 KL 损失、有效格子 mask；旧 thinking adapter 强制 thinking 标签，不直接套用。
5. 增加整盘评估器、分组报告和错误图；随后才做小规模拟合及正式训练。目前仅规划。

现有参考文件：`src/sokoban_state_modeling/state/codec.py`、`engine/env.py`、`sft/verl_dataset.py`、`sft/thinking_trace.py`，以及 Pev 的 `pev/openjev.py`。VERL 版本与多模态兼容性需按项目锁文件验证，不将本地静态审计等同 GPU 训练已跑通。

外部依据：[Transformers 官方 Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。数据规模、LoRA 策略、目标门槛与推进顺序均为本方案建议。
