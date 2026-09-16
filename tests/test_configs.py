from pathlib import Path

import yaml


CONFIG_ROOT = Path(__file__).parents[1] / "configs"


def test_split_configs_cover_exact_manifest_ranges():
    expected = {
        "dataset_train.yaml": 50_000,
        "dataset_validation.yaml": 2_000,
        "dataset_test.yaml": 10_000,
    }
    for filename, count in expected.items():
        row = yaml.safe_load((CONFIG_ROOT / filename).read_text(encoding="utf-8"))
        assert row["expected_tasks"] == count
        assert row["agent_name"] == "single_turn_agent"
        assert row["shuffle"] is False


def test_reasoning_profiles_are_explicit_opposites():
    thinking = yaml.safe_load((CONFIG_ROOT / "reasoning_thinking.yaml").read_text(encoding="utf-8"))
    direct = yaml.safe_load((CONFIG_ROOT / "reasoning_non_thinking.yaml").read_text(encoding="utf-8"))
    assert thinking["reasoning_mode"] == "thinking"
    assert thinking["thinking_prompt_open"] is True
    assert thinking["apply_chat_template_kwargs"]["enable_thinking"] is True
    assert direct["reasoning_mode"] == "non_thinking"
    assert direct["thinking_prompt_open"] is False
    assert direct["apply_chat_template_kwargs"]["enable_thinking"] is False

    for path in CONFIG_ROOT.glob("dataset_*.yaml"):
        split = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "thinking_prompt_open" not in split


def test_formal_ppo_profiles_are_complete_and_differ_only_in_reasoning_fields():
    thinking = yaml.safe_load((CONFIG_ROOT / "ppo_qwen35_4b_thinking.yaml").read_text(encoding="utf-8"))
    direct = yaml.safe_load((CONFIG_ROOT / "ppo_qwen35_4b_non_thinking.yaml").read_text(encoding="utf-8"))

    for profile in (thinking, direct):
        assert profile["algorithm"]["adv_estimator"] == "gae"
        assert profile["algorithm"]["filter_groups"]["enable"] is False
        assert profile["data"]["custom_cls"]["name"] == "SokobanStateDataset"
        assert profile["data"]["shuffle"] is False
        assert profile["data"]["validation_shuffle"] is False
        assert profile["actor_rollout_ref"]["actor"]["strategy"] == "fsdp"
        assert profile["critic"]["strategy"] == "fsdp"
        assert profile["reward"]["reward_manager"]["name"] == "SokobanRewardManager"
        assert profile["trainer"]["v1"]["trainer_mode"] == "sync"
        assert profile["trainer"]["v1"]["sampler"]["sync_refill_failed_groups"] is False

    assert thinking["data"]["reasoning_mode"] == "thinking"
    assert thinking["data"]["thinking_prompt_open"] is True
    assert thinking["data"]["apply_chat_template_kwargs"]["enable_thinking"] is True
    assert direct["data"]["reasoning_mode"] == "non_thinking"
    assert direct["data"]["thinking_prompt_open"] is False
    assert direct["data"]["apply_chat_template_kwargs"]["enable_thinking"] is False


def test_formal_grpo_profiles_disable_critic_and_preserve_group_contract():
    profiles = {
        mode: yaml.safe_load((CONFIG_ROOT / f"grpo_qwen35_4b_{mode}.yaml").read_text(encoding="utf-8"))
        for mode in ("thinking", "non_thinking")
    }
    for mode, profile in profiles.items():
        assert profile["algorithm"]["adv_estimator"] == "grpo"
        assert profile["algorithm"]["norm_adv_by_std_in_grpo"] is True
        assert profile["algorithm"]["filter_groups"]["enable"] is False
        assert profile["critic"]["enable"] is False
        assert profile["actor_rollout_ref"]["actor"]["use_kl_loss"] is True
        assert profile["actor_rollout_ref"]["rollout"]["n"] >= 2
        assert profile["data"]["gen_batch_size"] == profile["data"]["train_batch_size"]
        assert profile["data"]["train_batch_size"] % profile["actor_rollout_ref"]["actor"][
            "ppo_mini_batch_size"
        ] == 0
        assert profile["trainer"]["v1"]["trainer_mode"] == "sync"
        assert profile["data"]["reasoning_mode"] == mode


def test_training_launcher_selects_algorithm_and_reasoning_profile():
    script = (CONFIG_ROOT.parent / "scripts" / "train_qwen35_4b.sh").read_text(encoding="utf-8")
    assert 'TRAIN_ALGORITHM="${TRAIN_ALGORITHM:-ppo}"' in script
    assert "ppo|grpo" in script
    assert "${TRAIN_ALGORITHM}_qwen35_4b_${REASONING_MODE}.yaml" in script
