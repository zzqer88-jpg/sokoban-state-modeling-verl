"""Commit-gated VERL V1 sync trainer adapter.

Only the DataLoader policy and Sokoban metric projection differ from upstream.
The PPO update, GAE, checkpoint and weight-sync implementation stay inherited.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from sokoban_state_modeling.compat.verl_contract import verify_lock
from sokoban_state_modeling.data.sampler import FullCoverageSampler
from sokoban_state_modeling.env.verl_agent_loop import require_verified_verl_lock
from sokoban_state_modeling.training.metrics_hook import (
    aggregate_reward_columns,
    reward_columns_from_extra_fields,
)


_TRAINER_CLASS: type | None = None


def _require_runtime_checkout(lock: dict[str, Any]) -> Path:
    checkout_value = os.environ.get("VERL_CHECKOUT")
    if not checkout_value:
        raise RuntimeError("VERL_CHECKOUT must name the verified VERL checkout")
    checkout = Path(checkout_value).expanduser().resolve()
    lock_path = Path(os.environ.get("SOKOBAN_VERL_LOCK", "constraints/verl.lock.json")).resolve()
    observed = verify_lock(checkout, lock_path)
    if observed["commit"] != lock["commit"]:
        raise RuntimeError("VERL runtime and loaded lock disagree")
    return checkout


def _is_inside(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def build_sync_trainer_class():
    """Build the adapter only after the materialized VERL checkout passes its lock."""
    global _TRAINER_CLASS
    if _TRAINER_CLASS is not None:
        return _TRAINER_CLASS

    lock = require_verified_verl_lock()
    checkout = _require_runtime_checkout(lock)

    import transfer_queue as tq
    from omegaconf import OmegaConf, open_dict
    from torch.utils.data import SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    import verl
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
    from verl.utils.dataset.rl_dataset import collate_fn

    if not _is_inside(Path(verl.__file__), checkout):
        raise RuntimeError("the imported verl package does not come from VERL_CHECKOUT")

    class SokobanPPOTrainerSync(PPOTrainerSync):
        """Sync PPO with full dataset coverage and exact domain aggregation."""

        def _init_dataloader(self):
            if self.trainer_mode != "sync":
                raise RuntimeError("SokobanPPOTrainerSync only supports trainer.v1.trainer_mode=sync")
            filter_groups = self.config.algorithm.get("filter_groups", None)
            if filter_groups is not None and bool(filter_groups.get("enable", False)):
                raise RuntimeError("group filtering changes full-coverage semantics and must be disabled")
            if bool(self.config.trainer.v1.sampler.get("sync_refill_failed_groups", False)):
                raise RuntimeError("sync refill changes full-coverage semantics and must be disabled")
            if bool(self.config.data.get("shuffle", False)):
                raise RuntimeError("data.shuffle must be false; FullCoverageSampler owns ordering")
            if bool(self.config.data.get("validation_shuffle", False)):
                raise RuntimeError("data.validation_shuffle must be false")

            # Let upstream construct and validate both datasets, then replace only
            # the two loader objects and the schedule derived from loader length.
            super()._init_dataloader()
            gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
            if int(gen_batch_size) != int(self.config.data.train_batch_size):
                raise RuntimeError("the first sync profile requires gen_batch_size == train_batch_size")

            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=int(gen_batch_size),
                num_workers=int(self.config.data["dataloader_num_workers"]),
                drop_last=False,
                collate_fn=collate_fn,
                sampler=FullCoverageSampler(
                    self.train_dataset,
                    batch_size=int(gen_batch_size),
                    seed=int(self.config.data.get("seed", 0)),
                ),
            )
            self.train_dataloader_it = None
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=int(self.config.data.val_batch_size or len(self.val_dataset)),
                num_workers=int(self.config.data["dataloader_num_workers"]),
                drop_last=False,
                collate_fn=collate_fn,
                sampler=SequentialSampler(self.val_dataset),
            )

            self.steps_per_epoch = len(self.train_dataloader)
            total_training_steps = self.steps_per_epoch * int(self.config.trainer.total_epochs)
            if self.config.trainer.total_training_steps is not None:
                total_training_steps = int(self.config.trainer.total_training_steps)
            self.total_training_steps = total_training_steps
            optim_steps = total_training_steps * int(self.parameter_sync_step)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = optim_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = optim_steps

        def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
            metrics = super()._val_metrics_update(
                data_sources, sample_uids, reward_extra_infos_dict, sample_turns
            )
            metrics.update(aggregate_reward_columns(reward_extra_infos_dict, "val-sokoban"))
            return metrics

        def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
            super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
            values = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["extra_fields"],
            )
            extra_fields = list(values["extra_fields"])
            include = [
                not bool(extra_field["reward_extra_info"]["coverage_is_padding"])
                for extra_field in extra_fields
            ]
            all_columns = reward_columns_from_extra_fields(extra_fields)
            columns = reward_columns_from_extra_fields(extra_fields, include)
            metrics.update(aggregate_reward_columns(columns, "train-sokoban"))

            refs = [
                (int(cycle), int(position))
                for cycle, position in zip(
                    all_columns["coverage_cycle"], all_columns["coverage_position"], strict=True
                )
                if not math.isnan(cycle) and not math.isnan(position)
            ]
            if refs:
                cycle, position = max(refs)
                dataset_size = len(self.train_dataset)
                padded_size = len(self.train_dataloader) * int(self.config.data.train_batch_size)
                metrics["train-sokoban/coverage/unique_rollouts"] = (
                    cycle * dataset_size + min(position + 1, dataset_size)
                )
                metrics["train-sokoban/coverage/padding_rollouts"] = (
                    cycle * (padded_size - dataset_size) + max(0, position + 1 - dataset_size)
                )

    SokobanPPOTrainerSync.__name__ = "SokobanPPOTrainerSync"
    SokobanPPOTrainerSync.__qualname__ = "SokobanPPOTrainerSync"
    _TRAINER_CLASS = SokobanPPOTrainerSync
    return _TRAINER_CLASS
