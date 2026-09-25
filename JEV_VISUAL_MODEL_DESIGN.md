# Jev 视觉七分类模型改造设计

2026-09-25 更新：数据转换、jev 模块、本地多模态概率读出、KL 训练入口、checkpoint 和评估代码已实现；未启动正式训练，未修改 Pev。模型可见内容全部使用英文。操作说明见 [JEV_USAGE.md](JEV_USAGE.md)。

## 当前实施决策

继续在 `sokoban-state-modeling-verl` 中开发，不新建第三个完整项目，也不修改 Pev。新增 `src/sokoban_state_modeling/jev/`，仿照 Jev 文件命名，使用 Transformers/PyTorch 完成视觉候选概率训练和本地批量推理。当前阶段不安装、不依赖、不部署 SGLang；后续确有服务化需求时，才在固定版本的 openjev-sglang 独立分支扩展视觉服务。

确定路径为：完整图片 + 英文坐标问题 + 七类描述 → 完整多模态 Qwen3.5-4B → 原 LM head 的七个候选分数 → 格内 softmax → 七类概率。以概率分布计算 KL 监督，不新增 Pev ChoicePolicy，不要求生成答案或概率 JSON 文本。

当前源数据位于远程项目的 `data/sftcoldstartv11/sft/`；原始目标规模为训练 12,000、验证 1,200、测试 3,000 条。保持原 split，转换和去重后的规模以 audit.json 为准。本地尚未接入远程数据，因此未宣称已完成真实数据转换。

## 1. 依据 Pev 区分两条路径

`pev/openjev.py:OpenJevTeacher` 发送 state + choice criteria，取得每候选概率。`scripts/baseline_qwen.py:NativeQwen` 则复用 Jev PromptCompiler，读取原生词表头中 label_ids 对应的分数。它检验完整提示与标签拼接后的 token 边界，而非只检查孤立 A–G 是否单 token。

另一条路径是 `pev/features.py:feature_prompts` 的 Decision 分支和每候选 Option 分支，冻结骨干得到 context [D] 与 options [K,D]，再由 `pev/policy.py:ChoicePolicy` 的共享投影、类型嵌入、set attention、标量打分头输出 K 类分布。默认小网络宽度 64、2 层、4 个头。这里 attention 在同一问题的候选之间运行，不是不同格子之间的 attention。默认 prefix critic 另需一个特征，只服务价值估计。

推荐保留 Jev 原生候选读出路径，迁移 Pev 的概率接口与 KL 监督，不把两套架构混称为同一个模型。若原样迁移 Pev actor，七类每格至少需要 1 context + 7 option = 8 条逻辑特征编码，25 格共 200 条（未计 critic），不是 Jev 的 25 个问题分支。可缓存/批处理，但不能把其成本当作 Jev 首 token 读出成本。

## 2. 推荐结构与张量

完整 Qwen3.5-4B checkpoint + 配套 processor（图像编码器、连接模块、语言模型、原 LM head 均保留）。每个坐标构造独立图文分支：整图 + 棋盘尺寸/坐标说明 + 该格问题与七类英文描述。分支输入不能包含目标概率或任何真值棋盘。

若有效格子总数为 M、词表大小为 V：

1. processor 生成图文输入、位置元数据与 attention mask，按实际模型接口前向。
2. 在每个分支提示最后有效 token 的位置读取用于预测下一个 token 的 logits；不能读 padding 位置，不能把正确答案先喂进去再在答案后取 logits。
3. 按该分支实际 label_ids gather 七个分数，得到 z [M,7]。实现可以优化只对所需隐藏向量投影，但必须与完整 LM head 的结果一致，不能擅自假定词嵌入与 LM head 权重绑定。
4. 在七类维度做 FP32 log_softmax，得到 log_p；exp 后为 p [M,7]。输出再按 sample_id/row/column 还原为逐盘 [H,W,7]。
5. 七类全有效，不使用真值排除某些类别；变长棋盘的 cell_mask 与候选 valid mask 是不同概念。

不生成概率数字字符串，不使用全词表答案 token 交叉熵，也不运行动作策略、critic、PPO。类别是固定英文键，但字母标签的排列/映射由编译器产生；训练器必须按键名把 target 对齐到实际 label_ids。

## 3. 概率监督与训练接口

转换器 `records.jsonl` 的 input 进入模型，target.answers 进入损失，provenance 只进入日志。图片路径相对数据导出目录解析。训练 collator 为每个格子保存 board_index、row、column、label_ids、target_probs [7]，以及 batch 内每盘完整有效格子数。

复用 `pev/policy.py:teacher_kl` 的校验与 KL(q||p) 公式：q 有限、非负、键集合一致、七类和为 1。q 为引擎 one-hot；若引入教师软标签另标来源。实现稳定形式为 sum(xlogy(q,q) - q*log_p)。原 teacher_kl 对 batch 行直接平均；此处先按棋盘平均再跨棋盘平均，避免 8×8 比 6×6 权重大。若一盘拆成多个微批次，按其完整格子数加权后累积梯度，不能对每个微批次分别等权平均造成偏差。

可以沿用现有训练框架的分布式/优化器能力，但现有 thinking SFT dataset 的 assistant 字符串和 token loss 不能直接使用。需要新的数据 adapter 和自定义候选概率 loss；具体 VERL 扩展位置须核对项目锁定版本源码后确定，不凭名称臆造回调 API。简化的独立 PyTorch trainer 可以作为正确性参照，不必为了最初验证启动整套 RL。

用户已确定在服务器执行 full SFT：视觉编码器、连接模块、语言模型和 LM head 全部参与训练。LoRA 仅保留为可选模式。Pev `FrozenEncoder`/`Features` 强制 detach，并缓存冻结特征，不得直接套到要求 LoRA 反向传播的前向路径。所有被训练参数都必须出现在优化器里并检查实际梯度。

## 4. 后续可选的服务端改动（不属于第一阶段）

第一阶段的 service.py 是本地批处理协调器，不启动 HTTP 服务，不执行 SGLang 前缀预热。以下内容仅在后续决定接入远程服务时实施，以已审计的 openjev-sglang 提交 f3e1678168b2e9a298bb47639444b428b661c4b2 为基线：

| 位置 | 所需改动 |
|---|---|
| models.py / 请求协议 | 增加明确版本的图片输入契约；保持 questions/criteria 和 answers/probabilities 语义；不能把数据集整条记录直接发给服务 |
| prompts.py | 解除仅文本限制，使用图文 processor/template；共享图片 state、逐坐标 suffix；坐标写 instructions；保存实际标签映射 |
| backend.py | generate 除 input_ids 外传递后端支持的真实多模态载荷和元数据；保持每分支一次七标签概率读取 |
| service.py | warmup 与分支使用一致图文上下文；保留先预热后并发流程；图像 token 必须纳入长度/预算校验 |
| runtime.py / 模型加载 | 选择支持实际多模态 checkpoint 的后端配置；不能盲用目前 35B MoE profile 的参数加载 4B dense；确认适配器加载或合并方式 |
| 缓存 | 图片内容哈希、processor/模型/适配器版本及模板参与缓存语义；不同图片同问题不能串缓存；实际载荷以目标 SGLang 版本接口核对为准 |

首轮用无需共享的独立图文分支作为数值参照，再启用前缀复用并比较候选分布。SGLang 推理缓存不能用于训练反向传播。视觉 encoder 缓存与语言前缀缓存分别测量；API N+1 次调用不等于视觉只编码一次。

64 个问题上限意味着 6×6、7×7、8×8 单盘可装入一个请求，但仍受总输入预算限制。数据转换器的 request_groups 仅解决问题数，不保证图像 token 的总预算一定满足。

## 5. 正确性验收顺序

1. 图片与状态渲染对齐；每格七类 target 总和为 1，英文问题含正确坐标，输入无监督信息。
2. 模型与 Jev 编译的类别映射一致，实际答案边界是所选单 token；单问与多问批处理概率数值一致（允许定义好的浮点容差）。
3. target 为 one-hot 时，KL 等于七候选交叉熵；软 target 时与 Pev teacher_kl 同义；验证不同棋盘大小和微批拆分权重。
4. 可训练模块收到梯度；小数据能拟合。不能以代码通过静态检查宣称模型学会识别。
5. 同坐标换图得到对应预测；不同图片不错误命中缓存；优化缓存前后分布一致。
6. 使用独立地图验证整盘准确率和七类指标。训练后 HF 与部署后端候选概率对齐，检查 adapter 合并/加载是否改变输出。

## 6. 复用文件和新模块建议

复用概念/代码：Pev 的候选键校验、概率校验、KL 公式、NativeQwen 的实际 token 边界检查。扩展而非照搬：图文问题编译、模型 loader、概率训练器。无需搬入：ALFWorld Decision 动作语义、环境 rollout、PPO、critic、冻结文本特征缓存。

统一采用以下目录，替代先前分散到 models/sft/training 的新模块命名建议。以下文件已实现：

```text
src/sokoban_state_modeling/jev/
├── __init__.py
├── models.py       # 请求、choice 问题、概率响应等数据结构
├── prompts.py      # 英文图文问题编译、类别与标签 token 映射
├── backend.py      # 完整 Qwen/processor 加载与可微候选分数读取
├── scoring.py      # 概率校验、候选归一化、逐棋盘 KL 损失
├── dataset.py      # 图片与监督概率加载、坐标展开和 collator
├── training.py     # 优化、验证、checkpoint 与恢复
├── service.py      # 本地多坐标批处理和整盘结果汇总
├── evaluation.py   # 整盘/逐类指标、分组统计与错误样本
├── config.py       # 模型、数据、LoRA、批量和训练配置校验
└── cli.py          # train / evaluate / predict 命令入口
```

`models.py` 沿用 Jev 的命名含义，定义接口数据模型，不放神经网络主体。`backend.py` 的训练前向保留梯度，推理调用由上层明确进入 inference_mode；不能为了服务推理而把整个 backend 固定成 no_grad。`scoring.py` 是训练与推理共享的归一化来源，避免部署时额外温度造成分布不一致。`service.py` 不读取 target，`dataset.py` 将模型输入与监督目标分开。

现有 `sft/jev_export.py` 和 `scripts/export_jev_sft.py` 继续负责转换，不复制引擎、renderer 或转换器。现有 thinking SFT 入口保留原语义，新增独立配置如 `configs/jev_visual_sft.yaml`，不通过替换旧数据路径强行复用文本 loss。

模型保存目录必须记录基础 checkpoint 标识/版本、适配器权重、processor 设置、模板版本、类别顺序、实际 label_ids 和训练配置。恢复训练另存优化器、调度器、步数与随机数状态；预测加载时检查配置一致性，不能悄悄重新排列类别。

## 7. 实施阶段与交付标准

| 阶段 | 工作 | 完成标准 |
|---|---|---|
| 1 数据 | 完善已有转换器，对远程全部 split 做转换与审计 | PNG/标签对应，去重与泄漏检查通过，实际数量和七类统计可查 |
| 2 契约与编译 | models/config/prompts/dataset | 英文问题与坐标正确，目标不泄漏，候选 token 在完整答案边界有效，训练推理共用模板 |
| 3 概率模型 | backend/scoring | 单问题和批处理候选概率一致，归一化正确，KL 与参考公式一致，padding/微批次不改变预期权重 |
| 4 训练链路 | training 与 train CLI | 参数冻结和梯度检查通过，保存/恢复可验证；正式训练保持未启动，另列未来小样本拟合验证步骤 |
| 5 本地评估推理 | service/evaluation 与 evaluate/predict CLI | 按坐标还原完整概率棋盘，输出整盘准确率、七类指标、尺寸分组和错误样本 |
| 6 可选服务部署 | 另一个仓库中的视觉 Jev 分支 | 本地模型已验证后再启动；检查服务概率、并发与缓存一致性及实测性能 |

第一阶段交付包括数据转换命令（已有）、独立的 train/evaluate/predict 命令、配置示例、使用文档和必要测试。CLI 已实现并提供 audit/train/evaluate/predict 入口；只有显式 train 命令启动训练。服务器默认 full SFT，完整多模态模型参数均可训练；不依据假定 GPU 数量启动任务。

最低测试集覆盖：七类映射与键序对齐、非法概率、单问/批处理、不同尺寸及 padding、真实多模态输入路径、微批损失权重、梯度到达训练参数、checkpoint 重载一致性。无 GPU 时完成可运行的 CPU 单元测试，真实模型/GPU 验证单独报告状态，不以 mock 测试代替。

测试集仅用于最终报告；验证集用于选 checkpoint。当前只做代码与非训练检查；小样本拟合及正式训练均需在后续用户要求开始训练时执行。SGLang 的 64 问题限制只约束后续兼容服务；本地推理可按显存分批，不把数据中的 request_groups 当作模型架构限制。


## 8. 实现核对与范围（2026-09-25）

- 当前训练器明确是单进程 PyTorch 参考实现，不是 VERL 分布式训练器。配置默认 full SFT。可选 LoRA 模式只适配语言 full-attention 的 q/k/v/o 投影，不适配全部线性注意力层。
- 视觉连接模块以 `train_merger` 显式开关控制，默认 false；true 时使用 PEFT modules_to_save 保存其训练权重，已通过小模型梯度与重载检查。视觉塔始终存在。
- 使用左 padding 并校验最后位置有效，调用 logits_to_keep=1，避免生成整段序列的词表 logits。用实际完整提示检查标签 token 边界。
- 不实现推理图像缓存，因此不存在当前已实现图片只编码一次的承诺。SGLang HTTP/共享前缀是后续可选范围。
- 每 epoch 保存与恢复 optimizer/scheduler/RNG；恢复要求配置与数据摘要一致，新输出目录。可恢复已完成 epoch 的 checkpoint，不提供中途微批精确恢复或任意变更训练计划。
- 官方 processor 配合小型随机初始化 Qwen 已通过 CPU 集成测试，包含 linear/full attention、真实图片预处理、单问/批量、梯度与 checkpoint 概率一致性。没有下载或训练 4B 权重；GPU 与识别准确率未验证。


### 训练方式与本地验证边界

`finetune_mode` 支持 `full` / `lora`。full 训练完整多模态模型所有参数，包括视觉编码器、连接模块、语言模型与 LM head；不创建 LoRA，保存完整 model 权重。lora 使用适配器，并支持可选 train_merger。概率输出和 KL 目标与训练方式无关，不能因开发机仅有 CPU 就默认要求使用 LoRA。用户已选定服务器默认 full SFT；配置起始学习率为 1e-5。

本地仅下载官方 tokenizer、processor 和配置等小文件，没有下载 Qwen3.5-4B 权重。CPU 验证使用小尺寸随机初始化 Qwen（包括 hybrid attention），仅执行数值/梯度/保存恢复测试；原始 Sokoban 数据及正式 4B 训练均在远程服务器执行。单进程参考训练器需要足够单卡显存；多卡 FSDP 等不在当前已实现能力之中。


## 9. 单机 8×H800 full SFT（2026-09-25）

用户已明确服务器有 8 张 H800。新增 `jev/distributed.py`，采用 FSDP2（full-only）、BF16 计算、FP32 主参数/优化器分片与梯度规约；单进程路径保留作对照。配置为 `configs/jev_visual_sft_8xh800.yaml`，启动为 `scripts/train_jev_8xh800.sh`。本节替代上文“多卡未实现”的历史范围描述。

每卡默认每更新 1 张棋盘，全局 8 张；每卡每前向 4 个坐标问题。对不同棋盘尺寸和尾批，使用等长调度及零权重补齐，FSDP 平均梯度前按 world_size/真实棋盘数校正；真实棋盘不重复计入训练损失和验证指标。每 epoch 存分片训练状态与每卡 RNG，并导出完整模型，支持同卡数断点恢复及普通单进程推理。

CPU 测试覆盖调度无重复/遗漏、不同格子数与尾批的梯度等价、指标汇总。实际 FSDP2/NCCL 尚未在本地验证；服务器需先运行 `scripts/probe_jev_fsdp.py` 的微型随机 Qwen 检查，再开展正式 4B 实验。硬件配置不等于已测显存/吞吐。本实现初始化时每 rank 在 CPU 加载一份 FP32 权重，需规划服务器内存；未来可优化为 meta/rank-zero 初始化。

操作命令、恢复限制与资源说明见 [JEV_USAGE.md](JEV_USAGE.md) 最后一节。
