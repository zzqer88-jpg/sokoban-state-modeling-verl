"""Guarded PPO entry point for the standalone project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from pprint import pprint

from .trainer_sync import build_sync_trainer_class


def _task_runner_class():
    import ray

    @ray.remote
    class SokobanTaskRunnerV1:
        def __init__(self):
            self.config = None
            self.trainer = None
            self.agent_loop_manager = None

        def init_agent_loop_manager(self):
            from verl.trainer.ppo.v1 import AgentLoopManagerTQ
            from verl.utils.import_utils import load_class_from_fqn

            manager_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
                "agent_loop_manager_class"
            )
            manager_cls = (
                load_class_from_fqn(manager_fqn, "AgentLoopManager")
                if manager_fqn
                else AgentLoopManagerTQ
            )
            self.agent_loop_manager = manager_cls.create(
                config=self.config,
                llm_client=self.trainer.get_llm_client(),
                teacher_client=self.trainer.get_teacher_client(),
                reward_loop_worker_handles=self.trainer.get_reward_handles(),
            )

        def run(self, config):
            import transfer_queue as tq
            from omegaconf import OmegaConf
            from verl.utils.logging_utils import configure_verl_logging

            configure_verl_logging()
            trainer_cls = build_sync_trainer_class()
            if config.trainer.v1.trainer_mode != "sync":
                raise RuntimeError("the standalone baseline only enables VERL V1 sync PPO")
            config.transfer_queue.enable = True
            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)
            self.config = config
            tq.init(config.transfer_queue)
            succeeded = False
            try:
                self.trainer = trainer_cls(config=config)
                self.trainer.init()
                self.init_agent_loop_manager()
                self.trainer.fit(self.agent_loop_manager)
                succeeded = True
            finally:
                try:
                    tracking = getattr(self.trainer, "logger", None)
                    if tracking is not None:
                        tracking.finish(exit_code=0 if succeeded else 1)
                finally:
                    tq.close()

    return SokobanTaskRunnerV1


def launch(config) -> None:
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    build_sync_trainer_class()  # driver-side fail-fast lock/import check
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=_task_runner_class())


def compose_project_config(
    config_dir: Path,
    config_name: str,
    project_config: Path,
    overrides: list[str] | None = None,
):
    """Compose locked VERL defaults, project overrides, then CLI overrides."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(config_name=config_name)
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(project_config.resolve()))
    OmegaConf.set_struct(config, True)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compose the locked VERL ppo_trainer config and launch Sokoban sync PPO"
    )
    parser.add_argument("--config-name", default="ppo_trainer")
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--project-config", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path)
    args, overrides = parser.parse_known_args(argv)

    checkout_value = os.environ.get("VERL_CHECKOUT")
    if not checkout_value:
        parser.error("VERL_CHECKOUT is required")
    checkout = Path(checkout_value)
    project_root = Path(__file__).resolve().parents[3]
    os.environ.setdefault("SOKOBAN_VERL_LOCK", str(project_root / "constraints" / "verl.lock.json"))
    config_dir = (args.config_dir or checkout / "verl" / "trainer" / "config").resolve()
    if not config_dir.is_dir():
        parser.error(f"VERL config directory does not exist: {config_dir}")

    from omegaconf import OmegaConf

    project_config = args.project_config.expanduser().resolve()
    if not project_config.is_file():
        parser.error(f"project config does not exist: {project_config}")
    unsupported = [value for value in overrides if value.startswith(("+", "~")) or "=" not in value]
    if unsupported:
        parser.error(
            "project profiles accept only existing-key KEY=VALUE overrides; "
            f"unsupported overrides: {unsupported}"
        )
    config = compose_project_config(config_dir, args.config_name, project_config, overrides)
    if args.resolved_config:
        args.resolved_config.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, args.resolved_config, resolve=True)
    launch(config)


if __name__ == "__main__":
    main()
