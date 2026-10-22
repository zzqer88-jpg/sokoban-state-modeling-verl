# Perception-only thinking SFT

本版本只识别当前棋盘，保留 thinking。System prompt 保持原文：
`You are a precise Sokoban visual state-transition model.`
User prompt 不包含动作、动作规则或 prediction 要求。

监督目标为：

```text
Initial: player=(3,2); boxes=[(3,3)]; targets=[(3,4)].</think><perception>
######
#____#
#PXO_#
#____#
#____#
######
</perception>
```

坐标为 1-based `(row,column)`，目标点包含玩家或箱子覆盖的目标。
`<think>` 由模型 chat template 打开，target 不重复打开。
推演由当前状态确定性生成，不涉及动作执行。

## 在服务器生成

先拉取代码，设置项目路径，并安装生成所需依赖：

```bash
git pull --ff-only
export SOKOBAN_PROJECT_ROOT="$PWD"
export PYTHONPATH="$SOKOBAN_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install numpy Pillow PyYAML
```

已有原始 manifest 时，直接导出，无需重新生成棋盘或下载模型：

```bash
for split in train validation test; do
  python -m sokoban_state_modeling.scripts.export_perception_sft \
    --source "data/sft_cold_start/manifests/${split}.jsonl" \
    --output "data/perception_sft/${split}.perception.thinking.jsonl"
done
```

`--source` 也可以指向旧的 `train.thinking.jsonl` 等 SFT 文件。无需 `.idx`。
导出器只使用当前状态，不复制 query_actions、next_state、事件或 prediction。
同一当前棋盘可能对应多组旧动作，默认按 current_state_hash 去重；
因此导出数量可能少于原始 12,000/1,200/3,000，实际数量见 audit。

若没有棋盘数据，可先复用现有生成器，再运行上述导出：

```bash
python -m sokoban_state_modeling.scripts.generate_dataset --output-dir data/sft_cold_start --split train --num-tasks 12000 --resume
python -m sokoban_state_modeling.scripts.generate_dataset --output-dir data/sft_cold_start --split validation --num-tasks 1200 --resume
python -m sokoban_state_modeling.scripts.generate_dataset --output-dir data/sft_cold_start --split test --num-tasks 3000 --resume
python -m sokoban_state_modeling.scripts.audit_splits \
  --train data/sft_cold_start/manifests/train.jsonl \
  --validation data/sft_cold_start/manifests/validation.jsonl \
  --test data/sft_cold_start/manifests/test.jsonl \
  --output data/sft_cold_start/audits/split_leakage.json
```

该棋盘生成器仍会构造旧转移字段，但新导出器不使用它们。
也可提供只含 `current_state.grid` 与 `split` 的 JSONL；这时调用者必须保证
跨 split 的地图隔离。单次导出拒绝混合 split，但不检查其他文件的跨 split 泄漏。

## 导出契约与训练接入

- schema：`sokoban-perception-thinking-sft-v1`。
- messages 只有 system/user；加载器追加 `target` 作为 assistant。
- enable_thinking=true，thinking_prompt_open=true。
- 图片从 current_state 动态渲染，不把符号答案作为输入文本。
- 文件旁生成 `.audit.json`：源/输出 SHA-256、数量、去重数量、尺寸分布。
- 拒绝覆盖输出；失败保留 `.partial` 供排查，不发布未验证数据。
- 同一输入重复导出时，数据文件内容一致。

现有 `SokobanThinkingSFTDataset` 已增加该 schema 的校验与加载支持，
旧的双网格 schema 继续可用。训练使用该自定义 Dataset，并保持
`data.apply_chat_template_kwargs.enable_thinking=true`。

注意：原 `train_sft_qwen35_4b.sh` 中的 `preflight_sft` 专门检查旧双网格
协议和精确配额，不能直接用于本数据。此提交提供数据导出及 Dataset 接入，
不改变旧训练启动器或旧双网格 RL reward/parser。服务器的 tokenizer/GPU
训练仍需单独验证。
