# Sokoban State Modeling 独立 VERL 项目迁移计划

## 1. 目标

将当前 `sokoban_state_modeling` 从 VAGEN 仓库中的实验模块迁移为一个可单独安装、测试、生成数据和训练的项目。

最终项目允许复制并修改 VAGEN、RAGEN 和 `gym-sokoban` 中适用的 MIT 许可代码，但运行时不得依赖 `vagen` Python 包，也不得通过 VAGEN 的训练入口启动。

最终训练链路为：

```text
Sokoban manifest
    -> 独立 Dataset 动态渲染 RGB 与构造 prompt
    -> VERL 原生 rollout
    -> 独立规则奖励函数
    -> VERL 原生 PPO/GAE + FSDP
    -> VERL checkpoint / validation / logging
```

第一版固定范围：

- 任务：单轮 Sokoban 视觉状态建模；
- 模型：Qwen3.5-4B；
- 推理模式：显式支持 `thinking` 与 `non_thinking` 两个可复现实验分支，不使用隐式默认值；
- 输出协议：`<perception>...</perception><prediction>...</prediction>`；
- 奖励：当前确定性 dense 或 exact-only 规则奖励；
- 优势估计：VERL 原生 `gae`；
- critic：启用；
- trainer：先使用 VERL V1 `sync`；
- 训练后端：第一版固定 FSDP；FSDP2 只在同步基线通过后作为兼容性优化项测试；
- rollout：单轮，一个输入对应一个 trajectory；
- 不安装、不导入、不调用 VAGEN。

“第一版单轮”只限定当前 `DATASET_PLAN.md` 对应的训练任务，不删除多轮能力。项目从一开始保留独立的 Sokoban 状态机和 episode/session 边界；后续多轮训练由 VERL Agent Loop 调度该环境，不恢复 VAGEN 运行时依赖。

## 2. 当前状态

当前 `sokoban_state_modeling` 已经拥有大部分领域实现：

- RAGEN 风格 Sokoban 状态生成与渲染；
- manifest、索引和审计；
- 状态编码、动作执行与合法性检查；
- prompt 和严格 parser；
- dense/exact-only 奖励及指标；
- 数据生成、离线评估和单元测试。

目前仍存在以下 VAGEN 耦合：

1. `generator/dataset_builder.py` 从 VAGEN 导入 `set_seed`；
2. `env/sokoban_state_env.py` 继承 VAGEN 的 `GymImageEnv`；
3. 训练脚本调用 `python -m vagen.training.main`；
4. 训练脚本读取 `vagen/configs/training_defaults.flags`；
5. 训练数据使用 `vagen/training/dataset.py`；
6. 环境通过 `vagen/configs/env_registry.yaml` 注册；
7. 测试使用 VAGEN 的 `GymEnvAdapter` 和 `build_env`；
8. sampler、指标透传和日志聚合依赖 VAGEN trainer 的定制逻辑；
9. README 和 `DATASET_PLAN.md` 仍把 VAGEN 描述为运行时基础设施。

这些依赖需要全部替换或删除。

## 3. 目标目录结构

新仓库暂定名为 `sokoban-state-modeling-verl`，采用 `src` layout：

```text
sokoban-state-modeling-verl/
├── pyproject.toml
├── README.md
├── DATASET_PLAN.md
├── LICENSE
├── NOTICE
├── .gitignore
├── src/
│   └── sokoban_state_modeling/
│       ├── __init__.py
│       ├── engine/
│       ├── generator/
│       ├── state/
│       ├── env/
│       │   ├── core.py
│       │   ├── history.py
│       │   ├── session.py
│       │   └── verl_agent_loop.py
│       ├── prompt/
│       ├── reward/
│       │   ├── reasoning_adapter.py
│       │   ├── verl_reward.py
│       │   └── verl_reward_manager.py
│       ├── data/
│       │   ├── dataset.py
│       │   └── sampler.py
│       ├── training/
│       │   ├── main_ppo.py
│       │   ├── trainer_sync.py
│       │   └── metrics_hook.py
│       ├── assets/
│       │   └── sprites/
│       ├── compat/
│       │   └── verl_contract.py
│       └── utils/
│           └── seeding.py
├── constraints/
│   ├── verl.lock.json
│   └── runtime.lock.txt
├── configs/
│   ├── ppo_qwen35_4b_base.yaml
│   ├── ppo_qwen35_4b_thinking.yaml
│   ├── ppo_qwen35_4b_non_thinking.yaml
│   └── reward.yaml
├── scripts/
│   ├── generate_dataset.py
│   ├── audit_splits.py
│   ├── evaluate.py
│   ├── preflight_training.py
│   └── train_qwen35_4b.sh
├── tests/
└── outputs/
    └── .gitignore
```

manifest 和训练输出默认不进入 Git。

## 4. 准备修改的内容

### 4.1 建立独立包与依赖

新增 `pyproject.toml`，至少声明：

- Python 版本范围；
- NumPy；
- Pillow；
- PyYAML/OmegaConf 等数据和配置依赖；
- 测试 extra；
- VERL 的固定源码 commit；
- 与该 commit 实测兼容的 Python、PyTorch、Transformers、rollout backend 和 CUDA 版本锁。

正式运行不依赖 `gym-sokoban` 包：当前使用的七张 sprite 和必要 renderer 代码复制到本项目并作为 package data 安装，`gym-sokoban` 仅放入 migration-test extra，用于迁移期像素对照。这样不会因 PyPI 包缺失、版本元数据异常或 surface 目录变化而改变训练输入。

VERL 必须固定到经过验证的 commit，不能在正式训练中直接跟踪移动的 `main`。`constraints/verl.lock.json` 至少记录 repository URL、commit SHA、配置 schema 摘要和 custom Dataset/reward/dataloader 接口指纹；`runtime.lock.txt` 固定完整 Python/CUDA/torch/transformers/SGLang 或 vLLM 组合。当前仓库的 `verl` 只是尚未物化的 gitlink，指向 `JamesKrW/verl@c2cf4851f3f840eaca827b6069368242222f0218`，只能作为兼容性候选，不能在未 checkout、未跑 probe 的情况下宣称可用。

所有 probe 和训练入口必须要求 `VERL_CHECKOUT` 指向一个已物化的绝对目录，并核验其 remote URL、HEAD commit、工作树状态和 lock 中的接口 hash。该目录可以来自可用的 submodule，也可以是对锁定 repository/commit 的独立 clone；不得通过空 gitlink、当前 shell 的偶然 `PYTHONPATH` 或另一个同名 pip 包继续运行。

### 4.2 复制并独立化环境代码

保留当前以下领域模块的语义：

- `engine/`
- `generator/`
- `state/`
- `prompt/`
- `reward/`

对复制自 VAGEN/RAGEN/gym-sokoban 的文件：

- 保留来源说明；
- 标注来源仓库和 commit；
- 保留适用的版权与许可证声明；
- 在 `NOTICE` 中集中列出来源。

### 4.3 替换 VAGEN seeding

新增独立的 `utils/seeding.py`，复制并测试当前需要的随机状态保护逻辑：

```python
@contextmanager
def set_seed(seed):
    ...
```

`generator/dataset_builder.py` 改为只从本项目导入该实现。

验收要求：相同 seed 下生成结果、manifest hash 和渲染结果与迁移前一致。

### 4.4 移除 VAGEN 环境基类，保留独立多轮环境边界

多轮训练不要求依赖 VAGEN。VAGEN 的 `GymImageEnv`、`GymEnvAdapter`、环境注册表和多轮 runner 只是旧集成层；新项目保留 Sokoban 环境能力，但把它拆成领域层和 VERL 适配层：

1. `env/core.py` 是纯 Sokoban 状态机，负责 reset、primitive action、终止判断、渲染输入和确定性状态快照；不得导入 VAGEN 或 VERL；
2. `env/session.py` 管理单条 trajectory 的 episode 状态、turn 计数、最大步数、异常、snapshot/restore 和 close；不得包含 trainer 逻辑；
3. `env/verl_agent_loop.py` 是唯一允许导入 VERL 的环境适配层。后续实现 `SokobanAgentLoop(AgentLoopBase)`，调用 VERL 的 LLM client 生成、解析动作、调用 `SokobanSession.step()`、把新的文字/图像观察送回下一轮，并构造 VERL 需要的 trajectory；
4. 如果锁定 VERL 的 stateful `BaseTool` 生命周期比自定义 AgentLoop 更适合，则只把第 3 层替换为 `SokobanTool` adapter；`core.py` 和 `session.py` 接口保持不变。选择必须由阶段 0 的接口 probe 和多模态回传测试决定。

当前 `DATASET_PLAN.md` 定义的仍是“给定 1-3 个动作、一次回答最终状态”的单轮任务，所以第一版训练路径保持简单：

- Dataset 读取 manifest，调用同一个 core/renderer 构造初始 RGB 和 prompt；
- VERL `single_turn_agent` 生成一次回答；
- reward function 解析回答并用同一个 core 的真值比较，返回终局标量奖励和指标。

后续多轮任务使用 VERL Agent Loop，而不是 VAGEN runner：Dataset 为该 profile 提供 `agent_name=sokoban_agent` 和初始任务字段；每个 rollout 创建独立 `SokobanSession`；环境/tool observation token 的 `response_mask=0`，只有模型生成 token 参与策略损失；轨迹结束时释放 session。每轮图像、token 拼接、chat template、reward placement 和 checkpoint/resume 都必须针对锁定 VERL commit 做 contract test。

多轮模型输入必须保留完整的**可观察历史 trajectory**，不能只给最新一帧。第 `t` 轮生成动作前，模型可见上下文固定为：

```text
H_t = instruction, O_0, A_0, O_1, A_1, ..., O_t
```

- `O_0` 是初始真实 RGB；`O_i (i>0)` 是环境执行 `A_{i-1}` 后由同一 renderer 现场生成的真实 RGB；
- `A_i` 是模型该轮实际输出并被环境解析的 assistant turn，至少保留规范化 action payload；thinking profile 第一版保留实际 assistant message，包括该轮 reasoning，不能在 rollout 与训练两侧采用不同历史；
- observation 不公开隐藏真值、reward、`valid/noop/push`、可行动作集合或未来状态。非法/no-op 动作后 `O_{i+1}` 与 `O_i` 像素相同，但失败动作仍留在历史中，由模型结合前后图片判断；
- 历史中的图片和动作必须严格按时间顺序一一对应，不能重排、去重相同图片或用模型预测图替代环境图；
- 第一版不得做滑动窗口、历史摘要或仅最新帧模式。若完整 `H_t` 将超过锁定的 text/image/context budget，应在下一轮生成前按明确的 `context_limit` 终止 episode 并单独计数，不能静默丢弃早期 trajectory；
- Agent Loop 输出必须保存可重放的 `observation_hashes`、canonical actions、turn offsets、image-placeholder 映射和逐 token `response_mask`。环境 observation/image token mask 为 0，模型实际生成的 reasoning/action token mask 为 1。

多轮 replay 测试必须从同一初始 snapshot 顺序执行 action history，逐轮验证 state hash、原始 RGB hash 和 terminal 状态完全一致；还必须验证第 `t` 次 model generate 真正收到的 token IDs、图片序列和 `H_t` 定义一致。

必须避免两套真值：生成器、当前单轮 reward、多轮 session 和离线 replay 都调用同一个 `env/core.py`/`engine/` transition 实现。原 `env/sokoban_state_env.py` 不原样保留；其中有用逻辑迁移到上述独立接口，VAGEN adapter/registry 代码删除。

注意，未来“模型自主选择动作并多轮解 Sokoban”会改变 D01 的任务边界，不能静默复用当前 D01-D63 数据合同。它应建立单独的 multi-turn 数据/profile 规范和验收表；可以复用地图、状态编码、renderer 和 engine，但不得把多轮样本混入当前 50,000/2,000/10,000 split。

### 4.5 新增 VERL custom Dataset

新增 `src/sokoban_state_modeling/data/dataset.py`。

每个样本应至少提供：

- `data_source`；
- `raw_prompt`；
- 当前状态的 PIL RGB 图片；
- `task_id` 和 `task_index`；
- `query_actions`；
- 奖励评分需要的 current/next state；
- reward profile；
- VERL reward manager 需要的 ground truth/extra info。

Dataset 必须：

- 通过 manifest 的 `.idx` 做随机访问；
- 不保存冗余 PNG；
- 每次使用本项目 renderer 确定性渲染；
- 验证 manifest schema/generator/renderer 版本；
- 正确构造多模态 prompt；
- 能被 VERL `data.custom_cls` 直接加载。

对锁定 VERL commit，Dataset 构造函数固定兼容：

```python
SokobanStateDataset(
    data_files,
    tokenizer,
    config,
    processor=None,
    max_samples=-1,
)
```

`__getitem__` 必须返回锁定接口要求的完整 schema，不能凭旧版本猜字段。2026-09-15 观察到的 VERL `main` 将 chat-template/tokenization 移入 Agent Loop，因此当前候选形态是：结构化 `raw_prompt` 内含真实 PIL image、`data_source`、`reward_model.ground_truth`、固定 `extra_info` 和用于非空 batch 的 `dummy_tensor`；不要求 Dataset 预先生成 `input_ids/attention_mask/position_ids`。若最终锁定 commit 仍采用旧 tokenized Dataset 合同，则只在 compat adapter 中补这些 tensor，不污染领域 Dataset。项目增加 contract test，使用锁定 VERL 的 `collate_fn` 对真实样本组 batch，并送入一次实际 Agent Loop/rollout preprocessing；字段缺失、多出不支持字段或图片被重复 processor 都必须失败。

### 4.6 新增 VERL reward function

新增 `src/sokoban_state_modeling/reward/verl_reward.py`，提供 VERL 可加载的 `compute_score`。

它将调用现有：

- `parse_response()`；
- `score_response()`；
- `aggregate_metrics()` 所需的同名指标。

函数签名固定为锁定 VERL 的公开接口：

```python
compute_score(data_source, solution_str, ground_truth, extra_info=None)
```

返回 dict，至少包含训练主分数 `score`。锁定前必须确认该 VERL 版本会将 dict 的其余键透传为 reward extra info；确认后同时返回：

- 格式正确率；
- perception exact/F1；
- prediction exact/F1；
- joint success；
- transition consistency；
- conditional prediction numerator/denominator；
- scoring error 指标。

每个样本必须返回完全相同的数值 keyset，不能因 parse 失败、某个事件不存在或 thinking 未闭合而漏键：非法模型输出的业务指标填 0；基础设施异常对应的业务指标填 NaN，并增加固定的 error counter。conditional 指标只返回 numerator 和 denominator，不返回供上游直接求平均的逐样本 ratio。

阶段 0 还必须使用真实 tokenizer、chat template 和锁定 reward manager 检查传给 `compute_score` 的 `solution_str`：thinking 模式下必须仍能无歧义地找到 reasoning/answer 边界。若锁定栈以 `skip_special_tokens=True` 解码并吞掉 `</think>`，则启用项目内 `reward/verl_reward_manager.py`：它只负责从 response token IDs 做保边界解码、调用同一个 `compute_score`、再按锁定 VERL 的 batch 协议返回 reward tensor/extra info；不得靠文本形态猜测边界，也不得复制或修改 PPO 优化循环。该 manager 必须通过与 stock manager 的非 thinking 数值等价测试，并通过 thinking token-ID golden test 后才可进入正式配置。

不再依赖 VAGEN adapter 做指标透传或聚合。

### 4.7 替换 Dataset/Sampler 接入

不再使用 `vagen/training/dataset.py` 和 VAGEN 修改过的 sampler factory。

锁定 VERL V1 的 `trainer.v1.sampler.custom_sampler` 是 rollout ReplayBuffer 扩展点，不是 PyTorch DataLoader sampler，不能把当前 `FullCoverageSampler` 接到这里；stock V1 `_init_dataloader()` 固定 `drop_last=True`，也不符合 `DATASET_PLAN.md`。

第一版增加一个项目内的最小 VERL adapter：

1. `training/trainer_sync.py` 定义 `SokobanPPOTrainerSync(PPOTrainerSync)`；允许的定制面只有 DataLoader 初始化和领域指标归约/记录两个接口，具体 hook 名称由阶段 0 对锁定 commit 的 contract probe 决定；其余 PPO、GAE、critic、rollout、validation、checkpoint 和权重同步全部继承锁定 VERL；
2. 覆盖方法继续调用 VERL 的 `create_rl_dataset`、`StatefulDataLoader` 和 `collate_fn`，只把 train sampler 换成本项目 `FullCoverageSampler`，并明确设置 `drop_last=False`；validation 使用 `SequentialSampler`、`drop_last=False`、`validation_shuffle=False`；
3. `training/main_ppo.py` 是薄 Hydra/Ray 入口，复用 VERL 的 config validation、runtime env 和 TaskRunner 生命周期，但实例化上述 sync trainer；不得复制 VERL 的优化循环；
4. `FullCoverageSampler` 保持 50,000 unique + 48 tail padding 的 50,048 长 cycle，并在 cycle 结束后使用 seed+cycle 重新打乱；
5. sampler 不再只产出无法区分重复原因的裸 task index，而是产出可序列化的 `CoverageRef(task_index, cycle, position, is_padding)`；Dataset 将这些字段放入 `extra_info`，使 padding 与真实重复可逐样本审计；
6. StatefulDataLoader checkpoint 必须保存 sampler 的 cycle/permutation/position 和累计 coverage counters，任意位置恢复后 `CoverageRef` 序列逐项一致；
7. 400 steps × 128 的实际序列必须是完整 cycle 0 的 50,048 项，再接 cycle 1 的前 1,152 项；coverage metrics 读取 `CoverageRef`，不从 batch 数反推；
8. `training/metrics_hook.py` 只消费 reward extra info 和 `CoverageRef`，训练与 validation 都按固定 schema 记录总计数、分子和分母，再计算 `sum(numerator) / sum(denominator)`；分母为 0 时 rate 记为 N/A/不写数值，但必须保留 numerator=0、denominator=0；严禁 mean-of-ratios；
9. 同一 hook 生成按 k、event、size、boxes、terminated 的切片指标，并保留 scoring error、thinking closure/truncation/violation、padding/coverage counters。分布式归约必须先 all-reduce 原始 sum/count，再派生比率，不能先在 rank 内求比率再平均。

这是为了满足领域数据合同而增加的 VERL 集成薄层，不是复制 VAGEN trainer，也不修改 VERL 源码。adapter 必须为上游 `_init_dataloader()` 及阶段 0 选定的指标 hook 设置源码 hash/签名兼容门：任一接口发生变化时先失败并要求重新审计，不能静默运行。若锁定 commit 没有可复用的指标 hook，只允许在该薄 trainer 中截取 reward extra info 后调用 `metrics_hook.py`；不得复制完整 `fit()` 或优化器 step。async trainer 的 DataLoader、TransferQueue、ReplayBuffer 和指标传递语义不同，阶段 F 必须分别适配和验收。

### 4.8 改为 VERL 原生训练入口

重写 `scripts/train_qwen35_4b.sh`，不再读取任何 VAGEN 配置，也不再调用 VAGEN main。

目标入口：

```bash
python -m sokoban_state_modeling.training.main_ppo \
  --config-path="$PROJECT/configs" \
  --config-name="ppo_qwen35_4b_${REASONING_MODE}" \
  trainer.use_v1=true \
  trainer.v1.trainer_mode=sync \
  model_engine=dp \
  actor_rollout_ref.actor.strategy=fsdp \
  critic.strategy=fsdp \
  data.shuffle=false \
  algorithm.adv_estimator=gae \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  critic.enable=true
```

配置需要独立声明：

- model 和 processor；
- 第一版固定的 FSDP actor/critic 参数；
- SGLang rollout；
- prompt/response 长度；
- batch 和 micro-batch；
- optimizer；
- validation；
- checkpoint；
- WandB；
- custom Dataset；
- custom reward function；
- train/validation manifest。

三个项目 YAML 必须通过 Hydra defaults 继承锁定 VERL 的 `ppo_trainer`，并以显式必填的 `VERL_CHECKOUT` 构造绝对 `file://` search path 指向锁定 checkout。launcher 先验证目录/commit/hash，再执行 `--cfg job --resolve`，把完整 resolved config 写入 run 目录并运行 schema probe；环境变量缺失、路径未物化、未知键、缺少 custom class name、train Dataset 长度不是 50,000、`data.shuffle!=false` 或 `model_engine/strategy` 与锁定值不符都在启动 Ray 前失败。FSDP2 不与第一版混写在同一个配置中，另建实验配置并重新跑兼容门。

另外必须提供两套显式 reasoning profile，共享同一个 base 配置：

| profile | chat template | rollout reasoning | 答案协议 |
|---|---|---|---|
| `thinking` | `enable_thinking=true` | 使用锁定版本实测可用的 Qwen3.5 模板；第一版只依赖总 response 上限，不假定存在可强制闭合的 thinking budget | reasoning channel 结束后输出严格的 `<perception>/<prediction>` |
| `non_thinking` | `enable_thinking=false` | 不启用 reasoning parser/grammar，不生成可训练的 reasoning span | 直接输出严格的 `<perception>/<prediction>` |

启动脚本必须要求 `REASONING_MODE=thinking` 或 `REASONING_MODE=non_thinking`，缺失或非法值直接退出，不能依赖 VERL、Transformers 或模型仓库的默认值。两个 profile 使用相同 manifest、reward、PPO/GAE、batch、seed 和训练步数；只有 chat-template/reasoning 开关以及由此必需的 response/context budget 可以不同。thinking smoke 至少跑 256 条，完整 `</think>` 后存在合法 answer payload 的比例必须达到 99%，否则停止正式训练并先处理预算/后端兼容问题。

### 4.9 更新 preflight

`preflight_training.py` 需要新增检查：

- 当前工作目录和 `PYTHONPATH` 不包含 VAGEN；
- `import vagen` 失败不影响项目运行；
- 独立项目可以 import；
- VERL 版本/commit 符合锁定版本；
- VERL custom Dataset、reward dict extra-info、StatefulDataLoader resume 和多模态 schema probe 全部通过；
- 独立 `env/core.py` 和 `env/session.py` 不导入 VERL/VAGEN；reset/step/snapshot/restore/release contract 通过；
- 为后续多轮记录 VERL `AgentLoopBase`/`AgentLoopOutput`、stateful tool lifecycle、`response_mask` 和逐轮 multi-modal data 的接口指纹与 probe 结果；
- reward manager 解码后的 thinking 边界可恢复；训练/validation 的固定 extra-info keyset、分布式 sum/count 归约和切片日志 probe 全部通过；
- FullCoverageSampler 的长度、unique/padding/cycle 边界和 stateful resume 与训练预算一致；
- Dataset 能读取首尾行；
- 图片 dtype、shape、通道数正确；
- reward smoke test 通过；
- CUDA、GPU 数量和 PyTorch 正常；
- 训练、验证 manifest 数量与 split 标签正确。

### 4.10 重写测试边界

删除测试中的：

- `vagen.envs.GymEnvAdapter`；
- `vagen.envs.build_env`；
- VAGEN registry 假设；
- VAGEN trainer/mixin 假设。

新增：

1. 无 VAGEN import 测试；
2. 独立安装测试；
3. manifest Dataset 测试；
4. 多模态 prompt/image 对齐测试；
5. VERL reward function 测试；
6. 固定 seed/hash/render 回归测试；
7. 数据全集覆盖和 checkpoint sampler 恢复测试；
8. 一个 mock rollout 到 reward 的端到端测试；
9. 一个真实 VERL 单步 PPO smoke test；
10. checkpoint 保存与恢复 smoke test。
11. thinking/non-thinking 两个 profile 的 chat-template token golden test；
12. thinking 模式真实 token-ID 解码及 reasoning/answer 分离测试、可选薄 reward manager 等价测试，以及 non-thinking 模式无 reasoning token 测试；
13. 两个模式使用同一任务时 answer payload 得到完全相同的 reward 语义测试；
14. reward extra-info 固定 keyset、非法输出/基础设施异常和稀疏 batch 测试；
15. 训练与 validation 的单/多 rank sum-count、conditional 反例和全部切片指标测试；
16. 纯环境 core 的 reset/step/terminal/snapshot/restore 确定性测试；
17. 多个 `SokobanSession` 并发隔离、turn limit、异常和 release 测试；
18. 锁定 VERL 的 AgentLoop/tool adapter、response mask、逐轮文本/图像 observation 和轨迹 replay 集成测试。

第 16-17 项随第一版实现独立环境边界；第 18 项属于阶段 E 的后续多轮扩展，不阻塞当前单轮发布。

## 5. 分阶段实施

### 阶段 0：冻结真实 VERL 接口

- 将候选 VERL commit 物化到显式 `VERL_CHECKOUT`（可用 submodule 或独立 clone），不从网页文档、空 gitlink 或移动的 `main` 猜接口；
- 运行 CPU contract probes：Hydra compose、custom Dataset constructor、multimodal `collate_fn`、custom reward dict、reward `solution_str` 解码边界、固定 extra-info keyset、训练/validation 指标 hook、AgentLoop/Tool 生命周期与输出 schema、resolved config schema；
- 运行最小 GPU probe：Qwen3.5-4B processor、FSDP actor+critic 构建、所选 rollout backend、thinking/non-thinking 各生成一条；
- 对 `_init_dataloader()`、选定的指标 hook、AgentLoop/Tool 多轮接口，以及必要时启用的 reward manager 接口记录源码 hash/签名；将通过的 repository/commit/runtime 版本写入两个 lock 文件；
- lock 产生前禁止编写最终训练 YAML，也禁止生成正式数据发布标记。

完成标准：`preflight --verify-lock` 在干净环境通过，且输出的接口指纹与 lock 完全一致。若当前 `JamesKrW/verl@c2cf485...` 与官方 `verl-project/verl` 候选都测试，则优先选择满足全部 probe、且不需要 VAGEN patch 的官方 commit；失败原因写入兼容报告，不凭仓库名称猜测。

### 阶段 A：复制与静态解耦

- 创建新仓库目录和 `pyproject.toml`；
- 复制领域代码和测试；
- 内置 seeding；
- 把环境逻辑整理为不依赖 VERL/VAGEN 的 `core.py` 和 `session.py`，单轮与未来多轮共用；
- 清除所有运行时代码中的 `vagen` import；
- 添加 LICENSE/NOTICE；
- 让数据生成、审计和离线评估在独立虚拟环境中通过。

完成标准：

```bash
python scripts/preflight_training.py --check-no-vagen-runtime
```

该检查必须解析 `src/` 的 Python AST、检查 `pyproject.toml` 的运行时/extra 依赖、训练入口与配置目标，并在未安装 VAGEN、父目录不在 `PYTHONPATH` 的隔离环境完成 import 和数据 smoke。测试中用于证明 `import vagen` 失败的负向 fixture 可以出现该字符串，不能被全文搜索误判；任一真实运行时 import、依赖项或 VAGEN module target 则失败。

### 阶段 B：VERL Dataset 与奖励接入

- 实现 custom Dataset；
- 实现 `compute_score`；
- 实现固定 keyset、token-ID thinking 边界 probe，以及必要时启用的薄 reward manager；
- 实现训练/validation 共用的 sum/count 与切片指标 hook；
- 使用固定回答做 rollout/reward smoke test；
- 验证所有 reward 指标和旧实现一致，并用多 rank fixture 验证 conditional 不是 mean-of-ratios。

完成标准：固定任务、固定回答的主奖励和逐项指标逐值一致。

### 阶段 C：VERL V1 同步训练

- 添加独立 PPO 配置和启动脚本；
- 使用原生 GAE 和 critic；
- 分别运行 thinking 与 non-thinking 单步训练；
- 分别运行 thinking 与 non-thinking 短训练；
- 验证 checkpoint 保存、恢复和 validation。

完成标准：不把 VAGEN 加入 `PYTHONPATH` 也能完成训练。

### 阶段 D：结果对齐

- 用相同 manifest、模型、采样参数和 seed 对比迁移前后；
- 对比 prompt token IDs、图片、生成结果、reward 和训练指标；
- 记录不能做到 bitwise 一致的部分及原因。

完成标准：环境真值、规则奖励和数据覆盖必须完全一致；浮点训练指标在声明的容差内一致。

### 阶段 E：VERL 多轮环境接入

在不改变当前单轮数据合同的前提下，另建 multi-turn profile 和规范：

- 完成 `SokobanSession` 的并发隔离、snapshot/restore、最大 turn 和 release 测试；
- 针对锁定 VERL 选择并实现 custom `SokobanAgentLoop` 或 stateful `SokobanTool` adapter；
- Dataset 通过 `agent_name` 选择 sokoban loop，禁止单轮样本被误路由；
- 每轮执行 model generate → 严格动作解析 → environment step → 新 observation；
- 每次 model generate 都输入完整 `H_t = O_0,A_0,...,O_t`，不得只输入最新 RGB；非法/no-op 动作及像素相同的连续帧也不得从历史去重；
- 保证 model token mask=1、环境/tool observation mask=0，并验证总 prompt/response 长度；
- 在下一轮生成前预检完整历史的 text/image/context budget；超限按 `context_limit` 终止并报告，不截断旧历史；
- 对文本观察和每轮 RGB 观察分别做单步、短轨迹、多 session 并发、reward、checkpoint/replay 测试；
- thinking/non-thinking 在每轮都显式配置，且不能让 reasoning 文本被当成动作或环境观察。

完成标准：不安装 VAGEN 时，锁定 VERL 能完成至少一条确定性多轮 Sokoban trajectory；每轮 model input 包含完整可观察历史；逐轮 state/image/action/reward 可重放，session 无串扰，训练 mask、image placeholder 与轨迹 token 逐项对齐。若锁定 commit 的多轮多模态回传 probe 失败，先允许同样保留完整历史的文本/符号观察 profile，RGB 多轮不得伪装成已支持。

### 阶段 F：异步训练

同步单轮和同步多轮各自稳定后，再分别测试适用的异步模式：

1. `trainer.v1.trainer_mode=colocate_async`；
2. `trainer.v1.trainer_mode=separate_async`。

需要额外验证：

- policy version/staleness；
- TransferQueue/ReplayBuffer 恢复；
- 多模态图片不在异步队列中丢失；
- reward 与 task identity 不错配；
- session identity、turn 顺序和 environment state 不错配；
- 吞吐提升是否抵消参数同步和资源拆分开销。

## 6. 第一版明确不做

- 不复制 VAGEN trainer；
- 不复制 `VagenPPOTrainer` 或 trainer mixin；
- 不复制 VAGEN Dataset；
- 不修改 VERL 内部源代码；
- 不实现 VAGEN `no_concat`/`compact`；
- 不实现 bi-level GAE、turn-level GAE 或 turn GSPO；
- 不实现 token-span reward placement；
- 第一版训练不启用多轮自主解 Sokoban，但必须实现并测试独立 `core/session` 边界，不能删除未来的 VERL Agent Loop 接口；
- 不一开始就启用 fully async；
- 不把旧 smoke/legacy 数据全部复制进新 Git 仓库。

## 7. 风险与应对

### VERL API 变化

AgentLoop、reward loop、V1 async 和 multimodal schema 仍可能变化。应固定 VERL commit，并把 Dataset/reward 接口测试纳入 CI。

### Qwen3.5 运行栈兼容

Qwen3.5 的 Transformers、线性注意力依赖和 rollout backend 组合变化快；FSDP/FSDP2 的 wrap、activation offload 和权重同步也不是任意版本都可用。第一版只批准通过阶段 0 的 FSDP 组合，activation offload 默认关闭；FSDP2 和其他 offload 选项逐项做单步、保存、恢复和权重同步测试后才能加入正式配置。

### 多模态字段不匹配

不同 VERL 版本可能使用 `image`、`images` 或不同的 `multi_modal_data` 结构。必须根据固定 commit 写适配，不依赖推测。

### 多轮多模态回传仍不稳定

VERL Agent Loop 已提供多轮和 multi-modal data 扩展点，但 Qwen3.5、chat template、每轮新增图片、token 拼接与训练侧 processor 对齐仍必须以锁定 commit 实测。阶段 E 分开验收文本观察和 RGB 观察；后者失败时不得回退到 VAGEN，也不得静默把旧图重复当新观察，而是锁定文本 profile 或更换经过验证的 VERL commit。

### 指标聚合语义变化

条件指标不能直接对逐行或逐 rank ratio 求平均。必须由项目指标 hook 收集固定 keyset，分布式汇总 numerator/denominator 后再相除；不能假定锁定 VERL 会自动以领域语义记录所有 reward extra info。

### Thinking 边界在解码时丢失

某些 reward manager 会使用跳过 special token 的解码方式；如果 `</think>` 被移除，单看 `solution_str` 无法可靠区分 reasoning 与最终答案。阶段 0 必须用 token-ID golden case 验证；失败时启用项目薄 reward manager 做保边界解码，禁止用“最后一个标签之前都是思考”等启发式恢复。

### 数据覆盖变化

第一版不接错 VERL 的 ReplayBuffer custom sampler。同步训练通过项目内 VERL trainer 薄适配层注入 FullCoverageSampler 和 `drop_last=False`，固定 coverage、padding 和 resume；async 另行验证。

### 隐式导入 VAGEN

只做 `rg` 不够。必须在一个没有安装 VAGEN、父目录也不在 `PYTHONPATH` 的隔离环境中执行 import、数据生成和测试。

## 8. 最终验收清单

- [ ] 新项目拥有自己的 `pyproject.toml`、README、LICENSE 和 NOTICE；
- [ ] 代码中无运行时 `vagen` import；
- [ ] 训练脚本不调用 `vagen.training.main`；
- [ ] VAGEN 不在安装依赖和 `PYTHONPATH` 中；
- [ ] 独立 Sokoban `core/session` 已保留，且不导入 VAGEN 或 VERL；单轮 reward 与多轮 session 共用同一 transition 真值；
- [ ] VERL commit/runtime lock 和接口指纹已经由阶段 0 probe 生成并验证；
- [ ] 第一版明确使用 FSDP，未混入未经验证的 FSDP2/offload 配置；
- [ ] 数据生成与审计测试通过；
- [ ] 固定 seed 的 manifest/hash/render 与迁移前一致；
- [ ] custom Dataset 正确返回文字、图片和 ground truth；
- [ ] VERL reward function 与旧评分逐值一致；
- [ ] reward manager 向 `compute_score` 提供的 thinking/answer 边界已用真实 token IDs 验证；必要时已启用并验收项目薄 reward manager；
- [ ] 训练和 validation 指标使用同一固定 keyset，conditional/切片指标以全局 sum/count 计算且通过多 rank 反例测试；
- [ ] thinking 与 non-thinking 配置均显式设置且不存在默认模式漂移；
- [ ] VERL V1 sync 在 thinking 与 non-thinking 下分别完成单步 PPO；
- [ ] thinking channel 与 answer payload 正确分离，两个模式共用完全相同的数据和答案评分规则；
- [ ] validation、checkpoint 和 resume 成功；
- [ ] FullCoverageSampler 的 51,200 个实际 rollout index 与 checkpoint position 可逐项重放；
- [ ] 正式 50,000/2,000/10,000 split 通过泄漏审计；

以下是后续扩展门，不阻塞第一版单轮发布：

- [ ] 单轮同步基线稳定后，独立 multi-turn profile 已通过 VERL AgentLoop/tool、mask、逐轮 observation 和 session 隔离 smoke；
- [ ] 第 `t` 轮输入严格包含 `O_0,A_0,...,O_t`，相同帧不去重、历史不静默截断，token/image/hash replay 与实际 rollout 一致；
- [ ] 单轮和多轮各自稳定后再批准对应异步 trainer。

## 9. 预期交付物

1. 独立的 `sokoban-state-modeling-verl` 源码目录；
2. 可复现的依赖和固定 VERL 版本；
3. 可直接执行的数据生成、审计、评估和训练脚本；
4. VERL 原生 Dataset 与 reward 接入；
5. 无 VAGEN 环境下的测试结果；
6. 迁移前后行为对齐报告；
7. 独立 Sokoban core/session 与后续 VERL AgentLoop/tool adapter；
8. 后续切换 VERL V1 async 的配置与验证记录。

## 10. `DATASET_PLAN.md` 完整迁移追踪

### 10.1 完整性的判定

本节把 `DATASET_PLAN.md` 视为新项目的数据与任务语义唯一规范。迁移完成不等于“相关模块已经复制”，而必须同时满足：

1. `DATASET_PLAN.md` 原文复制到新项目根目录，并记录源文件 SHA-256；
2. 第 1-12 节的每项约束都落到新项目代码、配置、测试或审计报告中；
3. 生成器、renderer、prompt、parser、reward 和 metrics 的行为与规范一致；
4. 第 10 节列出的 15 条强制验收条件全部自动化并通过；
5. 本追踪表没有 `未映射` 项；
6. 独立项目不因脱离 VAGEN 而放宽配额、回退策略、确定性或恢复语义。

本次复审所对应的源文件 SHA-256 为 `B0CAB7385CB436FB648E366474AA5252719F74DADCD1F23B99AF0A5513DBA09D`。开始实施时先复算；若不一致，必须重新生成章节覆盖索引和 D01-D63 追踪，不能沿用本次“无缺漏”结论。

`DATASET_PLAN.md` 第 13 节原先描述的是“实验留在 VAGEN 仓库”的接入方式，其中继续导入 `GymImageEnv`、VAGEN Dataset/trainer 和修改 VAGEN registry/mixin 的内容，与本次独立项目目标冲突。处理规则是：

- 第 1-12 节以及第 13 节中的领域模块职责、复制来源和算法约束全部保留；
- 第 13.4-13.6 节涉及 VAGEN 运行时接入的部分，由本文第 3-6 节的独立 package、VERL Dataset、VERL reward function 和原生训练入口等价替代；
- 这种替代只改变集成边界，不改变任何数据、状态、动作、渲染、prompt、reward、metrics、split 或覆盖语义。

### 10.2 任务与单条 manifest 契约

| 需求 ID | 必须原样保留的约束 | 新项目落点 | 强制验收 |
|---|---|---|---|
| D01 | 任务只做当前状态感知和执行 1-3 个给定 primitive actions 后的单个最终状态预测；不自主选动作、不求解、不输出中间轨迹 | `prompt/template.py`、VERL Dataset、reward | prompt golden test；输出协议 test；无 action-selection 字段 |
| D02 | manifest 为 JSONL，不保存 PNG；固定 `schema_version=sokoban-state-modeling-v2`、`generator_version=ragen-state-manifest-v9` | `generator/dataset_builder.py`、`manifest_store.py` | schema/version mismatch 必须失败 |
| D03 | 完整保存 task/split/seed/attempt/profile、resolved generator config、四类 hash、current/next 两层状态与 grid、actions、事件、执行数、终止和 metadata | builder schema + typed validator | golden manifest 逐字段测试；缺字段拒绝 |
| D04 | 模型可见内容只能是 current RGB、`query_actions`、固定说明和输出格式；所有真值及审计 metadata 必须隐藏 | `data/dataset.py` | Dataset public/private 字段泄漏测试 |
| D05 | `fixed` 编码为墙 0/地板 1/目标 2；`entity` 编码为空 0/玩家 1/箱子 2；复合字符仅 `# _ O X P * +` | `state/codec.py` | 双向 round-trip、冲突编码、非法字符测试 |
| D06 | 数组和复合 grid 必须同时存储且逐格一致；shape/dtype/长度边界明确 | codec + manifest validator | 任一表示篡改即加载失败 |
| D07 | 可评分 answer payload 严格为 `<perception>` 后接 `<prediction>`，不含 `<reasoning>`、`<answer>` 或自然语言参考答案；模型原生 thinking channel 不属于 answer payload | template/parser/reasoning adapter | golden prompt、answer extraction 和 forbidden-tag test |

### 10.3 当前状态直接生成与分布契约

| 需求 ID | 必须原样保留的约束 | 新项目落点 | 强制验收 |
|---|---|---|---|
| D08 | 当前状态从 reverse-search trajectory 直接取快照，禁止以 `prefix_actions` 回放构造 | `engine/utils.py`、`generator/sampler.py` | manifest/schema 中禁止 prefix；状态来源测试 |
| D09 | main-v1 固定 seed 18427；尺寸 6/7/8 权重 60/25/15；箱数 1/2 权重 90/10 | sampler profile YAML + sampler | 50k 精确整数配额审计 |
| D10 | topology multiplier 0.75/1/1.25/1.5 为 15/50/25/10；基准 `floor(1.7*(H+W))`；Decimal half-up；固定值与采样互斥；记录 resolved integer | sampler | 边界值/舍入 golden test |
| D11 | `p_change_directions` 0.15/0.35/0.60/0.80 为 15/50/25/10；reverse depth 100/300/500 为 20/60/20；progress near/middle/far 为 20/55/25 | sampler | profile 精确配额与 resolved config 审计 |
| D12 | generation seed 公式为 `dataset_seed + split_id*10,000,000 + task_index`；attempt 使用 `SeedSequence([generation_seed, attempt])` + PCG64；不得使用未播种全局 RNG | seeding + builder | 跨进程重复生成 byte-identical；记录 NumPy/bit-generator 版本 |
| D13 | reverse snapshot 每深度最多 16、按 `dmax` 三等分、目标桶最终最多 256 且稳定排序；目标桶为空整任务重采样，不 fallback | reverse search/sampler | reservoir 上限、稳定排序、空桶失败测试 |
| D14 | 玩家距离以“不推箱子的最短可行走距离到箱子接触位”定义；adjacent/near/medium/far 为 30/30/25/15；adjacent 内 pushable/blocked 为 70/30；空桶重生成 | sampler + validation | BFS 距离 golden maps；配额；无 fallback |
| D15 | interior wall ratio 范围 `[0,0.55]`；corridor ratio 按规范定义；派生分布只报告、不作为隐式拒绝条件 | sampler/audit | 公式单测与 audit histogram |
| D16 | 六个互斥主类别 ordinary/player_near/near_completion/box_on_target/player_on_target/narrow-high-wall 为 50/20/15/5/5/5，全部 solvable 且 current 非终局 | category sampler/validation | 谓词互斥、精确配额、非终局测试 |
| D17 | ordinary 排除所有特殊谓词；box/player on target 强制 2 箱；各类别使用规范中的精确定义 | category predicates | 每类正反例 fixture |
| D18 | 使用联合 contingency table 满足箱数、类别、距离、尺寸、topology、p-change、reverse-progress 的硬联动，不用彼此独立的 rejection sampling | quota planner | 联合表行列和及各硬联动审计 |
| D19 | far 全部 8x8，其中 topology 1.5 占 10%、1.25 占 5%，p-change=.35、ordinary；narrow 类为 topology=.75/p-change=.15/near-goal；其他 player/terminal/category anchors 按规范固定 | quota planner | 每条联动的零违规断言 |
| D20 | wall/corridor/fork/connectivity、最短解、reverse-depth ratio、box-target distance、重叠属性和整块不变只做派生报告 | audit | 报告字段完整性测试 |

### 10.4 动作块、事件与终止契约

| 需求 ID | 必须原样保留的约束 | 新项目落点 | 强制验收 |
|---|---|---|---|
| D21 | `k=1/2/3` 精确为 40/35/25；50k 对应 20,000/17,500/12,500，共 92,500 transitions | action quota planner | 整数配额审计 |
| D22 | 事件缺口权重为 move25/push20/wall-noop15/blocked-box-noop15/enter-target10/leave-target5/player-target-transition10 | `generator/action_sampler.py` | target 与实际分布同时报告 |
| D23 | 事件最低覆盖依次为 20/12/10/8/5/2/7%；事件内方向差 smoke <=10pp、正式集 <=8pp | action audit | 低于阈值或方向超差即失败 |
| D24 | primary event 优先级固定为 enter、leave、push、blocked-noop、player-target、wall-noop、move | transition classifier | 多事件动作 golden test |
| D25 | box-on-target、player-on-target、player-near pushable/blocked、terminal 的 rare-event anchors 按规范强制绑定 | quota/action sampler | 各 anchor 零缺失测试 |
| D26 | 约束优先级固定为：中间不终止/终局只在最后 > 长度配额 > rare anchors > event deficit > direction | action sampler | 冲突候选 deterministic test |
| D27 | 恰好 10% 的任务由最后一个动作通关；前 `k-1` 个动作绝不通关；所有动作都执行，`executed_action_count==k`，无通关后缀 | transition + action sampler | replay/early-terminal/terminal-quota 测试 |
| D28 | `next_state` 只表示整个动作块后的单个最终状态；通关本身不进入 reward | transition/reward | final-state replay + reward independence test |

### 10.5 求解、失败、唯一性与恢复契约

| 需求 ID | 必须原样保留的约束 | 新项目落点 | 强制验收 |
|---|---|---|---|
| D29 | forward BFS 的状态和动作展开确定性且方向顺序 U/D/L/R | solver/validation | 固定地图访问顺序与最短解 golden test |
| D30 | attempts=500、reverse nodes=200k、forward BFS nodes=500k、per-depth=16、bucket=256、每 map/state 各最多一条任务 | config + generator | 配置锁定与上限触发测试 |
| D31 | 区分 reverse depth reached 与 node-limit；solver unknown 丢弃；发布集全部 solvable | generator/audit | 三种 solver 状态计数；发布 unknown/unsolvable=0 |
| D32 | 尝试耗尽时整个 build 非零退出并记录失败桶；禁止 fallback、放宽约束、复制样本 | builder | 故障注入/非零退出/无部分正式发布测试 |
| D33 | split 内 map/current/task 唯一性及跨 split 泄漏规则完全按规范；变尺寸不 pad/resize | builder/audit/Dataset collate | 重复与跨 split 注入测试；变尺寸 batch 测试 |
| D34 | staging 每 100 条原子保存；仅相同 split/size/version/profile 可恢复；从最大完整 chunk 重算；正式发布后清 staging | builder | kill-and-resume byte-identical test |
| D35 | `.idx` 保存精确 byte offsets，`task_index` 等于连续行号；绝对规范路径缓存；首尾/随机 offset、路径和版本不符均 fatal | manifest store | 索引篡改和多 worker 随机读取测试 |

### 10.6 渲染、prompt、Dataset 与训练覆盖契约

| 需求 ID | 必须原样保留的约束 | 新项目落点 | 强制验收 |
|---|---|---|---|
| D36 | 复制并锁定 VAGEN 当前 native Sokoban renderer 与七张 sprite；项目 renderer 不做主题、噪声、增强、压缩、resize、pad、旋转或翻转；从 fixed+entity 现场生成 uint8 RGB。Qwen processor 必需的模型输入变换属于训练预处理，不能回写或替代原始 RGB，且其版本/参数必须锁定 | local renderer/assets + Dataset | renderer/sprite SHA-256、原始像素回归、dtype/shape/channel 全量审计；processor 输出 contract test |
| D37 | prompt 使用规范中的固定英文模板；actions 采用大小写固定、无换行的紧凑 JSON；单轮 stop string 为 `</prediction>` | template + VERL config | prompt byte/token golden test |
| D38 | Dataset 按 task index 读取已存真值，不在 rollout 时生成、求解或重算 next state；同一 task 重放的 RGB/actions/truth 完全相同 | custom Dataset | 同 index 多 worker 重放一致性测试 |
| D39 | train/validation/test 规模固定 50,000/2,000/10,000；train 尺寸 30,000/12,500/7,500，validation 667/667/666，test 3,334/3,333/3,333 | build profiles | exact-count audit |
| D40 | 训练预算固定 batch128、400 steps、rollout.n=1、51,200 episodes；前一 cycle 为 50,000 unique + 48 padding，共 50,048/391 batches，余下 1,152 来自下一 cycle | VERL config + sampler | budget/coverage counter test |
| D41 | 禁止 RandomSampler 和 `drop_last`；FullCoverageSampler 固化 cycle seed、permutation、position 和 tail padding，并产出带 `is_padding` 的 `CoverageRef`；由项目 VERL sync trainer 薄适配层以 StatefulDataLoader + `drop_last=False` 注入；恢复后序列逐项一致 | `data/sampler.py` + `training/trainer_sync.py` | 50,048 cycle、后续 1,152、逐样本 padding 标志、任意位置 resume property test、上游方法 hash guard |

### 10.7 Parser、reward 与指标契约

| 需求 ID | 必须原样保留的约束 | 新项目落点 | 强制验收 |
|---|---|---|---|
| D42 | domain parser 只接收 reasoning adapter 分离后的 answer payload，只把 CRLF 归一到 LF，不 strip；两段独立解析；标签恰一次且有序；严格 H 行/W 字符/字符集 | `prompt/parser.py` | 缺失、重复、乱序、额外文本、fence、CRLF、空行、空格、非法字符、尺寸参数化测试 |
| D43 | 一段错误只令该段状态分为 0；额外文本/错误标签顺序令 format_exact=0，但可解析段仍独立计分 | parser/reward | 组合错误 truth table |
| D44 | 语义合法性要求恰一玩家、箱/目标数匹配、实体不在墙上、边界墙；玩家受困仍合法；预测 fixed 可变化但必须自身合法 | parser/validation | 语义边界 fixture |
| D45 | per-class macro F1、empty-class、unchanged-delta 规则以及 dense 公式必须逐项保持：perception=.5 full+.5 exact；prediction=.25 full+.25 delta+.5 exact；总分=.4 perception+.5 prediction+.05 format+.05 exact_all，并 clamp | reward | 手算 fixture 与旧实现逐值一致 |
| D46 | exact-only 固定为 `.1 format + .4 perception_exact + .5 prediction_exact`；两个 profile 除 reward profile 外使用相同 run 配置 | reward/config | 两 profile golden test/config diff test |
| D47 | 不使用 LLM judge、自然语言关系奖励、求解奖励或 token/turn 分段 credit；只给 VERL 原生 GAE 一个单轮终局标量 | reward + PPO config | 静态配置检查和 reward API test |
| D48 | 固定 11 项指标及其精确定义；conditional 只保存 numerator/denominator，训练与 validation 先跨 batch/rank 求和再相除，分母 0 为 N/A 并保留计数 | `training/metrics_hook.py` + VERL extra info | 单/多 rank 聚合反例测试，禁止 mean-of-ratios |
| D49 | joint 不含 format；strict trajectory success 使用 exact_all；cell/F1 为 per-task macro；每个样本返回固定数值 keyset；非法模型输出为 0，基础设施异常为 NaN+error | reward manager + `training/metrics_hook.py` | 稀疏/失败 batch 聚合与故障注入测试 |
| D50 | 指标至少按 k、event、size、boxes、terminated 切片；切片也先聚合 sum/count 再派生比率 | `training/metrics_hook.py` + evaluation | 训练/validation 报告 schema 完整性测试 |

### 10.8 Thinking / Non-thinking 双模式契约

thinking 是模型生成策略，不是数据集变体。两个模式读取完全相同的 manifest、图片、prompt 任务正文、隐藏真值和 reward profile，因此不需要也不允许生成 `thinking` 专用数据集。

| 需求 ID | 必须满足的约束 | 新项目落点 | 强制验收 |
|---|---|---|---|
| D57 | 提供 `thinking` 和 `non_thinking` 两个显式 profile；启动时必须二选一，配置与日志必须记录 `reasoning_mode`，禁止依赖默认值 | `configs/ppo_qwen35_4b_*.yaml`、launcher、preflight | 缺少/非法模式失败；resolved config snapshot test |
| D58 | thinking 模式使用锁定版本实际支持的 Qwen3.5 chat-template 开关；non-thinking 明确关闭；processor、训练 tokenization 与 rollout engine 的模板参数必须一致 | Dataset/tokenizer/rollout config | 同一 mode 下 prompt token IDs 三方一致；模式切换 golden token diff |
| D59 | thinking response 必须先由独立 reasoning adapter 分成 `reasoning_content` 和 `answer_payload`，domain parser 只解析后者。必须兼容 Qwen chat template 将 opening `<think>` 放在 prompt、response 仅含 reasoning+`</think>` 的情况，也兼容 backend 返回完整 `<think>...</think>` 的情况；阶段 0 必须确认 reward 解码不丢边界，若丢失则用 token-ID 感知的薄 reward manager；不得启发式猜边界，也不得为容纳思考而放宽 D07、D42-D44 的答案格式 | `reward/reasoning_adapter.py`、`verl_reward.py`、必要时 `verl_reward_manager.py` | 真实 token-ID 解码、prompt-open/完整 block/空/未闭合/重复 close tag、answer 前后垃圾文本测试 |
| D60 | non-thinking response 不得包含非空 reasoning channel；若模型仍输出 `<think>`，按模式违规单独计数，answer payload 仍按同一严格规则评分 | reasoning adapter/metrics | non-thinking reasoning violation test |
| D61 | thinking 第一版只使用锁定的总 response/context 上限，不假定 stock VERL/SGLang 支持强制 thinking budget；未闭合思考或因上限耗尽没有完整 answer payload 时该样本答案无效并单独计数。256 条 smoke 的完整思考后答案率须 >=99%；若另加 backend limiter，必须先以 token-ID 和 logprob 对齐测试证明可用 | rollout config/preflight | 总预算算术、未闭合/截断计数、closure-rate 门禁；可选 limiter contract test |
| D62 | 两种模式共享同一个 dataset hash、split、sampler 顺序、reward 公式、GAE/critic、优化预算和评估集；不得把 reasoning 文本加入 manifest 或 ground truth | config validator/release metadata | 两个 resolved configs allowlist diff；dataset SHA-256 完全相同 |
| D63 | validation 至少报告 `reasoning_mode`、reasoning token 数、answer token 数、thinking closure/truncation/violation，并分别给出原模式和交叉模式评估 | evaluation/logging | 2x2 checkpoint×inference-mode 报告 schema 与计数测试 |

这里的 2x2 评估指：thinking 训练 checkpoint 分别以 thinking/non-thinking 推理，以及 non-thinking 训练 checkpoint 分别以 thinking/non-thinking 推理。它用于判断模型是否真的同时具备两种使用方式；主训练仍先采用两个独立实验分支，不在同一 batch 中随机混合模式。若以后要训练单个混合模式 checkpoint，应作为新实验 profile 单独设计和审计，不能静默改变第一版对照实验。

### 10.9 Hash split、审计和发布门禁

| 需求 ID | 必须原样保留的约束 | 新项目落点 | 强制验收 |
|---|---|---|---|
| D51 | 四类 SHA-256 使用规范中的字段、shape/dtype/version 和有长度边界的规范序列化；`map_hash` 绝不包含实体，`next_state_hash` 不参与 split | codec/hash module | 固定 byte-vector golden digest |
| D52 | digest 前 8 byte big-endian 得到 u；区间 train `[0,.61)`、validation `[.61,.73)`、test `[.73,1)`；先分 map 再生成 state/action | split planner | 三个边界值和 map ownership 测试 |
| D53 | 同一地图/状态的所有查询只能属于一个 split；跨 split map/current 零交集；split 内 task 去重；同 profile 仅以 map hash 隔离 | audit | 三 split 两两交集门禁 |
| D54 | 审计报告包含规范列出的任务/唯一 hash/重复/动作长度与方向/事件/类别属性/尺寸箱数/解长/solver/wall/target/noop/terminal/early-terminal/交集/replay/render/coverage 全部字段 | `generator/audit.py` | audit JSON schema required-fields test |
| D55 | `DATASET_PLAN.md` 第 10 节 15 条强制验收全部成为发布门禁，包括 golden response、复制/常量/常见效果基线、parser 边界和 sampler resume | release audit + tests | 任一失败不得生成 release marker |
| D56 | 正式 manifest、idx、audit、profile、源码 commit、依赖版本、generator/renderer/sprite hash 一并记录；版本变化禁止静默复用 | release metadata | provenance 完整性与 mismatch test |

### 10.10 从旧实现复制到独立项目的文件级映射

| 旧项目内容 | 新项目内容 | 迁移要求 |
|---|---|---|
| `engine/env.py`, `engine/utils.py` | `src/sokoban_state_modeling/engine/` | 连同 renderer/assets、solver、反向搜索一起复制；移除 VAGEN import，不改变 D08-D20、D29-D36 语义 |
| `state/*.py` | `src/sokoban_state_modeling/state/` | 保留 codec、hash、transition、validation 的全部约束与测试 |
| `generator/*.py` | `src/sokoban_state_modeling/generator/` | 连同 profile、quota planner、manifest/index、staging/resume 和 audit 迁移，不能只复制 builder 主路径 |
| `prompt/*.py` | `src/sokoban_state_modeling/prompt/` | 模板和 parser 一起迁移，保留精确字节与边界行为 |
| `reward/*.py` | `src/sokoban_state_modeling/reward/` | 保留现有纯评分层，新增薄的 `verl_reward.py` 与 `reasoning_adapter.py`；先分离 thinking channel，再调用同一个严格 parser，不在适配层重写公式；仅在 stock 解码丢失 thinking 边界时启用 `verl_reward_manager.py` |
| `training/sampler.py` | `src/sokoban_state_modeling/data/sampler.py` | 保留 state_dict/load_state_dict、permutation、unique-first、tail padding 和 counters；由项目 sync trainer 薄适配层接入 VERL StatefulDataLoader，不接入 ReplayBuffer custom sampler |
| 现有 reward/metrics 聚合逻辑 | `src/sokoban_state_modeling/training/metrics_hook.py` | 固定 extra-info keyset；训练与 validation 统一按全局 sum/count 派生 conditional 和切片指标；禁止依赖上游默认平均语义 |
| `env/sokoban_state_env.py` 中的领域逻辑 | `src/sokoban_state_modeling/env/core.py`、`session.py` | 移除 `GymImageEnv` 继承和 VAGEN adapter/registry；保留 reset/step/terminal/snapshot/release，调用唯一 engine 真值，并供单轮与多轮共用 |
| VAGEN 多轮 runner/adapter | `src/sokoban_state_modeling/env/verl_agent_loop.py` | 不复制运行时；按锁定 VERL 实现薄 `AgentLoopBase` 或 stateful `BaseTool` adapter，负责 session 编排、逐轮 observation 和 trajectory mask |
| `configs/*.yaml` | `configs/` | 复制数据与奖励 profile；训练字段改为锁定 VERL commit 的原生键；用测试保证非接入字段不漂移 |
| `scripts/*.py` | `scripts/` | 迁移 generate/audit/evaluate/preflight；train 脚本改为 VERL 原生入口 |
| `tests/*.py` | `tests/` | 保留所有领域测试，删除 VAGEN adapter/registry 假设，并增加 D01-D63 的独立集成与发布测试 |

### 10.11 实施时的门禁顺序

迁移不能直接以“PPO 能启动”作为完成信号，必须按以下顺序放行：

1. **接口冻结门**：checkout VERL 候选，probe 后写入 commit/runtime lock；
2. **合同冻结门**：复制 `DATASET_PLAN.md`，记录 hash，生成 D01-D63 的测试清单；
3. **领域等价门**：固定 fixtures 下 codec/hash/generator/render/prompt/parser/reward/metrics 与旧实现逐值或逐字节一致；
4. **数据 smoke 门**：1,000 条 smoke 全部通过审计和 15 条发布条件；
5. **VERL 接入门**：custom Dataset/reward/FullCoverageSampler 和 sync trainer 薄适配层在无 VAGEN 环境分别以 thinking/non-thinking 完成单步 PPO、validation 和 checkpoint resume；
6. **正式数据门**：50,000/2,000/10,000 三个 split 达到所有精确配额、覆盖率、唯一性和泄漏要求；
7. **同步训练门**：400 steps sync 训练完成，coverage counters 与预算完全一致；
8. **多轮接入门**：另建 multi-turn profile/规范，以锁定 VERL AgentLoop/tool 完成确定性 trajectory、完整 `O_0,A_0,...,O_t` 可观察历史、session 隔离、context-limit、mask、image-placeholder 和逐轮 observation 验收，不修改当前 D01-D63 单轮合同；
9. **异步试验门**：只有对应的同步单轮/多轮门通过后才评估 async，且 async 不得改变各自数据和环境合同。

每一门都应产出机器可读 JSON 报告。前七门及 D01-D63 全部通过即完成当前单轮迁移；第 8 门是后续多轮扩展，第 9 门是对应同步基线之后的可选异步扩展。进入任一后续门前，其依赖门的报告状态必须为 `passed` 且所有 required checks 为真。

### 10.12 源文档章节覆盖索引

| `DATASET_PLAN.md` 章节 | 迁移需求 | 覆盖状态 |
|---|---|---|
| 第 1 节 目标与边界 | D01、D04、D07、D28、D47 | 已映射 |
| 第 2 节 单条任务的数据格式 | D02-D07 | 已映射 |
| 第 3 节 状态表示 | D05-D06、D36 | 已映射 |
| 第 4 节 直接生成多样化当前状态 | D08-D20、D29-D33 | 已映射 |
| 第 5 节 动作块与事件 | D21-D28 | 已映射 |
| 第 6 节 渲染 | D36 | 已映射 |
| 第 7.1-7.3 节 reset/step、manifest store | D37-D38、D35 | 已映射；由 VERL Dataset/reward 取代 VAGEN Env 生命周期 |
| 第 7.4 节完整遍历 | D40-D41 | 已映射 |
| 第 7.5 节 parser | D42-D44 | 已映射 |
| 第 8.1-8.2 节 reward | D45-D47 | 已映射 |
| 第 8.3 节指标 | D48-D50 | 已映射 |
| 第 9 节规模与拆分 | D09、D16、D21、D27、D39-D41、D51-D53 | 已映射 |
| 第 10 节生成后审计 | D54-D56 | 已映射 |
| 第 11 节实施顺序 | 本文第 5 节及 10.11 | 已映射并增加 VERL 接口冻结和独立化前置门 |
| 第 12 节明确不做 | 本文第 6 节、D01、D08、D28、D47 | 已映射 |
| 第 13.1-13.3 节领域目录与职责 | 本文第 3-5 节及 10.10 | 已映射到 `src` layout |
| 第 13.4-13.6 节 VAGEN 接入 | 本文第 4.4-4.8 节 | 已替换为 VERL 原生 Dataset/reward/sampler/entrypoint，不改变数据语义 |
| 第 13.7 节完成条件 | 本文第 5、8、10.11 节 | 已映射并收紧为 D01-D63 全量门禁 |

## 11. 当前计划审计结论

在加入第 10 节之前，本文只覆盖了独立化架构、主要模块和训练入口，未逐项写明 profile 精确配额、联合约束、动作事件下限、失败不回退、原子恢复、hash split、parser 边界、reward 公式、11 项指标及 15 条发布门禁，因此不能称为完整迁移计划。

加入 D01-D63 后，`DATASET_PLAN.md` 第 1-12 节的需求已全部映射到独立项目落点和验收方式；第 13 节的 VAGEN 接入内容也已明确由 VERL 原生接口替代；thinking/non-thinking 作为训练和推理配置维度加入，但不改变数据集契约。未来多轮能力通过独立 `core/session` 加 VERL AgentLoop/tool adapter 实现，不依赖 VAGEN；由于自主多轮会改变 D01，必须使用单独 profile 和规范，不能混入当前单轮发布集。

2026-09-15 的可执行性复审另外修正了六项问题：不再把 V1 ReplayBuffer custom sampler 误当 DataLoader sampler，而是用项目内 sync trainer 薄适配层注入 FullCoverageSampler 与 `drop_last=False`；正确处理 Qwen prompt-open thinking；为 reward 解码丢失 reasoning/answer 边界准备 token-ID 感知的薄 manager；不再假定上游会按领域语义聚合 extra info，而是加入固定 keyset、全局 sum/count 和切片 metrics hook；把 renderer sprite 真正内置进 package；第一版从模糊的 FSDP/FSDP2 改为经过阶段 0 probe 后锁定 FSDP。迁移前 22 个已有单元测试全部通过，但它们只证明旧领域实现基线可用，不证明独立 VERL 接入已经完成。

因此本文现在是一份可按门禁执行的迁移计划，而不是一份无需验证即可直接启动正式训练的现成配置。阶段 0 必须先产生真实 VERL commit/runtime lock；在 lock 和 GPU probe 通过以前，不得声称 PPO 配置已可运行。实际实施时，前七道门和 D01-D63 必须全部通过后才能声明当前单轮迁移完成；未来多轮 profile 还必须通过第 8 门自己的环境/轨迹合同，第 9 门异步不作为第一版完成条件。

## 12. 本次可执行性复审依据

复审日期为 2026-09-15。外部链接只用于确认当前上游接口形态，正式实现只以阶段 0 锁定的源码 commit 为准：

- [VERL `main_ppo.py`](https://github.com/verl-project/verl/blob/main/verl/trainer/main_ppo.py)：V1 TaskRunner、config validation 和训练入口；
- [VERL `trainer_base.py`](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/v1/trainer_base.py)：custom Dataset 创建、StatefulDataLoader、stock `drop_last=True` 以及 ReplayBuffer custom sampler 的真实层级；
- [VERL `rl_dataset.py`](https://github.com/verl-project/verl/blob/main/verl/utils/dataset/rl_dataset.py)：`data.custom_cls.path/name` 加载协议；
- [VERL reward config](https://github.com/verl-project/verl/blob/main/verl/trainer/config/reward/reward.yaml) 与 [naive manager](https://github.com/verl-project/verl/blob/main/verl/experimental/reward_loop/reward_manager/naive.py)：`reward.custom_reward_function` 签名、dict `score` 与 extra-info 透传；
- [VERL PPO config](https://github.com/verl-project/verl/blob/main/verl/trainer/config/ppo_trainer.yaml)：`gae`、critic、FSDP、`sync/colocate_async/separate_async` 的当前配置结构；
- [VERL Agent Loop 文档](https://github.com/verl-project/verl/blob/main/docs/advance/agent_loop.rst)、[Agent Loop 实现](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/agent_loop.py) 与 [rollout 配置](https://github.com/verl-project/verl/blob/main/verl/trainer/config/rollout/rollout.yaml)：custom agent loop、`agent_name` 路由、stateful tool、多轮 token mask 和 multi-modal data 扩展点；
- [SGLang reasoning parser](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/parser/reasoning_parser.py)：prompt 预先打开 thinking channel 时，response 可能只有 reasoning 内容、`</think>` 与最终答案；
- 迁移前本地基线：`sokoban_state_modeling/tests` 共 22 项通过；当前旧仓库的 `verl` gitlink 未 checkout，因此没有进行本地 VERL import、Hydra compose 或 GPU smoke。

执行时必须把这些“当前 main 的观察”替换为 lock 中的精确文件 hash 和 contract probe 结果。只要任何上游接口与本文假设不一致，就在阶段 0 更新 adapter 和计划后再继续，不能跳过门禁。

## 13. 当前实施状态

截至 2026-09-15，已在独立目录完成第一批实现：

- 建立 `src` package、依赖元数据、LICENSE/NOTICE 和第三方许可证；
- 复制并解耦 engine/generator/state/prompt/reward，内置七张 sprite，替换 VAGEN seeding；
- 实现不依赖 VAGEN/VERL 的 `SokobanCore`、`SokobanSession` 和完整 `O_0,A_0,...,O_t` 历史结构；
- 实现 manifest Dataset 的领域 payload、带 `CoverageRef` 的 FullCoverageSampler、reasoning adapter、VERL reward 函数和 sum/count metrics accumulator；
- 增加显式 thinking/non-thinking 与 dataset/reward 配置；
- 实现无 VAGEN 静态 preflight 和 VERL checkout/lock 只读校验；
- 独立项目测试为 37 passed；20 条 train smoke 数据的 manifest 和 idx 与迁移前逐字节一致，独立 audit 通过。
- 数据生成热点已完成第一轮无损优化：把 wall/category 静态拒绝移到前向求解之前，并以 64-bit bitboard BFS 替换小数组 NumPy BFS；保留原 solver 作为 differential oracle。固定 30 条 train 基准从 11.041 秒降至 2.508 秒（约 4.4x），manifest 与 idx SHA-256 均逐字节不变。
- 已物化官方 VERL checkout 并把 `mll-lab-nu/RAGEN` 当前依赖方向与官方 VERL 主线分开处理；本项目锁定并源码审计 `verl-project/verl@61134f92099913b6f8fa5920ac5839a4a314212d`，记录文件及关键符号 hash。
- 已按该提交生成 sync Trainer 与独立 TaskRunner：保持 `trainer_mode=sync`，注入 FullCoverageSampler、`drop_last=false`、顺序 validation，以及训练/validation 共用的领域指标归约。

尚未完成且不得伪装为可用：

- 当前运行环境没有 PyTorch，尚未执行真实 torch Dataset/collate contract；
- VERL 源码接口已经锁定，但 CPU tokenizer/Hydra 和 GPU FSDP/rollout probe 未完成，因此 `constraints/verl.lock.json` 仍保持 `verified=false`；
- `training/trainer_sync.py` 与 `training/main_ppo.py` 已生成但受 lock 门禁保护；`reward/verl_reward_manager.py` 和多轮 `env/verl_agent_loop.py` 仍保持 fail-closed，分别等待真实 tokenizer 边界测试及后续多轮 profile；
- 正式 `ppo_qwen35_4b_*.yaml` 必须等 Stage 0 的 Hydra、FSDP、reward decode、AgentLoop 和 GPU probes 后生成。
