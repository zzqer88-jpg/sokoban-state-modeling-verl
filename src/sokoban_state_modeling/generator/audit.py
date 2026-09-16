"""Dataset manifest integrity and distribution audit."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import numpy as np

from sokoban_state_modeling import GENERATOR_VERSION, RENDERER_VERSION, SCHEMA_VERSION
from sokoban_state_modeling.engine import render_state, renderer_provenance
from sokoban_state_modeling.reward.reward import score_response
from .manifest_store import ManifestStore
from .sampler import MAIN_V1, QuotaSchedule
from sokoban_state_modeling.state.codec import grid_to_state, map_hash, state_hash, state_to_grid, task_hash
from sokoban_state_modeling.state.transition import apply_actions
from sokoban_state_modeling.state.validation import solve_forward, validation_errors


def audit_manifest(manifest_path: str | Path, index_path: str | Path | None = None) -> dict:
    store = ManifestStore(manifest_path, index_path)
    counts = {
        name: Counter() for name in (
            "dim_room", "num_boxes", "topology_multiplier", "p_change_directions",
            "reverse_search_depth", "reverse_progress", "player_distance",
            "adjacent_type", "query_length", "events", "primary_state_category",
            "terminated", "solver_status", "state_attributes",
        )
    }
    event_directions: dict[str, Counter] = {}
    hashes = {name: set() for name in ("map", "current", "task")}
    failures: list[str] = []
    baselines = {"copy_changed_count": 0, "copy_changed_dense_reward_sum": 0.0, "copy_no_change_count": 0}
    split: str | None = None
    for index in range(len(store)):
        try:
            row = store.read(index)
            current, nxt = grid_to_state(row["current_state"]["grid"]), grid_to_state(row["next_state"]["grid"])
            num_boxes = int(row["resolved_generator_config"]["num_boxes"])
            if validation_errors(current, num_boxes) or validation_errors(nxt, num_boxes):
                raise ValueError("invalid current or next state")
            if map_hash(current) != row["map_hash"] or state_hash(current) != row["current_state_hash"]:
                raise ValueError("current map/state hash mismatch")
            if state_hash(nxt) != row["next_state_hash"] or task_hash(current, row["query_actions"]) != row["task_hash"]:
                raise ValueError("next/task hash mismatch")
            replayed, events, subtypes, terminal = apply_actions(current, row["query_actions"])
            if state_hash(replayed) != row["next_state_hash"] or events != row["event_sequence"] or subtypes != row["event_subtype_sequence"] or terminal != row["terminated"]:
                raise ValueError("transition replay mismatch")
            rendered = render_state(current)
            expected_shape = (current.shape[0] * 16, current.shape[1] * 16, 3)
            if rendered.dtype != np.uint8 or rendered.shape != expected_shape:
                raise ValueError("online renderer returned an invalid RGB frame")
            for name, value in (("map", row["map_hash"]), ("current", row["current_state_hash"]), ("task", row["task_hash"])):
                if value in hashes[name]:
                    raise ValueError(f"duplicate {name} hash")
                hashes[name].add(value)
            cfg, metadata = row["resolved_generator_config"], row["metadata"]
            solver = solve_forward(current, MAIN_V1.max_forward_bfs_nodes)
            if solver.status != "solvable" or solver.shortest_steps != metadata["shortest_solution_steps"]:
                raise ValueError("solver status/shortest path mismatch")
            split = split or row["split"]
            if row["split"] != split:
                raise ValueError("one manifest cannot mix splits")
            counts["dim_room"][str(cfg["dim_room"])] += 1
            counts["num_boxes"][str(cfg["num_boxes"])] += 1
            counts["topology_multiplier"][str(cfg["topology_multiplier"])] += 1
            counts["p_change_directions"][str(cfg["p_change_directions"])] += 1
            counts["reverse_search_depth"][str(cfg["reverse_search_depth"])] += 1
            counts["reverse_progress"][cfg["reverse_progress_bucket"]] += 1
            counts["player_distance"][metadata["player_box_distance_bucket"]] += 1
            if metadata.get("adjacent_type"):
                counts["adjacent_type"][metadata["adjacent_type"]] += 1
            counts["query_length"][str(len(row["query_actions"]))] += 1
            counts["events"].update(row["event_sequence"])
            for event, direction in zip(row["event_sequence"], row["query_actions"]):
                event_directions.setdefault(event, Counter())[direction] += 1
            counts["primary_state_category"][metadata["primary_state_category"]] += 1
            counts["terminated"][str(row["terminated"])] += 1
            counts["solver_status"][metadata["solver_status"]] += 1
            for attribute, enabled in metadata.get("state_attributes", {}).items():
                if enabled:
                    counts["state_attributes"][attribute] += 1
            if row["executed_action_count"] != len(row["query_actions"]):
                raise ValueError("executed_action_count mismatch")
            category = metadata["primary_state_category"]
            required = {
                "box_on_target": {"box_leaves_target"},
                "player_on_target": {"player_target_transition"},
            }.get(category)
            if category == "player_near_box":
                required = {"blocked_box_noop"} if metadata["adjacent_type"] == "blocked_adjacent" else {"box_push", "box_enters_target", "box_leaves_target"}
            if row["terminated"]:
                required = {"box_enters_target"}
            if required and not any(event in required for event in events):
                raise ValueError(f"missing required anchor event for {category}")

            golden = "<perception>\n" + "\n".join(state_to_grid(current)) + "\n</perception>\n<prediction>\n" + "\n".join(state_to_grid(nxt)) + "\n</prediction>"
            for profile in ("dense", "exact_only"):
                if score_response(golden, current, nxt, row["query_actions"], num_boxes, profile).reward != 1.0:
                    raise ValueError(f"golden response is not perfect under {profile}")
            copy = "<perception>\n" + "\n".join(state_to_grid(current)) + "\n</perception>\n<prediction>\n" + "\n".join(state_to_grid(current)) + "\n</prediction>"
            copy_result = score_response(copy, current, nxt, row["query_actions"], num_boxes, "dense")
            if state_hash(current) != state_hash(nxt):
                baselines["copy_changed_count"] += 1
                baselines["copy_changed_dense_reward_sum"] += copy_result.reward
                if copy_result.metrics["prediction_success_rate"] or copy_result.metrics["changed_cell_f1"] or copy_result.components["prediction_score"] > 0.25:
                    raise ValueError("copy-current baseline received transition credit on a changed state")
            else:
                baselines["copy_no_change_count"] += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"row {index}: {exc}")
    violations: list[str] = []
    if not failures and split is not None and len(store) >= 20:
        expected_schedule = QuotaSchedule(
            len(store), MAIN_V1, MAIN_V1.dataset_seed + {"train": 0, "validation": 10_000_000, "test": 20_000_000}[split], split=split
        )
        expected_rows = [expected_schedule.assignment(i) for i in range(len(store))]
        comparisons = {
            "dim_room": Counter(str(list(row["dim_room"])) for row in expected_rows),
            "num_boxes": Counter(str(row["num_boxes"]) for row in expected_rows),
            "topology_multiplier": Counter(str(row["topology_multiplier"]) for row in expected_rows),
            "p_change_directions": Counter(str(row["p_change_directions"]) for row in expected_rows),
            "reverse_search_depth": Counter(str(row["reverse_search_depth"]) for row in expected_rows),
            "reverse_progress": Counter(row["reverse_progress"] for row in expected_rows),
            "player_distance": Counter(row["player_distance"] for row in expected_rows),
            "query_length": Counter(str(row["action_length"]) for row in expected_rows),
            "primary_state_category": Counter(row["primary_state_category"] for row in expected_rows),
            "terminated": Counter(str(bool(row["terminal"])) for row in expected_rows),
        }
        for name, expected in comparisons.items():
            if counts[name] != expected:
                violations.append(f"{name} quota mismatch: actual={dict(counts[name])}, expected={dict(expected)}")
        adjacent = sum(counts["adjacent_type"].values())
        expected_pushable = int(np.floor(0.70 * adjacent + 0.5))
        if counts["adjacent_type"]["pushable_adjacent"] != expected_pushable:
            violations.append("adjacent pushable/blocked 70/30 quota mismatch")
        event_total = sum(counts["events"].values())
        minimum_coverage = {
            "player_move": .20, "box_push": .12, "wall_noop": .10,
            "blocked_box_noop": .08, "box_enters_target": .05,
            "box_leaves_target": .02, "player_target_transition": .07,
        }
        if len(store) >= 1000:
            direction_tolerance = 0.10 if len(store) < 10_000 else 0.08
            for event, minimum in minimum_coverage.items():
                actual = counts["events"][event] / event_total if event_total else 0.0
                if actual < minimum:
                    violations.append(f"event {event} ratio {actual:.4f} is below coverage floor {minimum:.4f}")
                direction_counts = event_directions.get(event, Counter())
                ratios = [direction_counts[name] / max(1, sum(direction_counts.values())) for name in ("Up", "Down", "Left", "Right")]
                if sum(direction_counts.values()) >= 100 and max(ratios) - min(ratios) > direction_tolerance:
                    violations.append(f"event {event} direction spread exceeds {direction_tolerance:.2f}: {ratios}")
        if counts["solver_status"] != Counter({"solvable": len(store)}):
            violations.append("published tasks are not 100% solvable")
    resolved_manifest = Path(manifest_path).resolve()
    resolved_index = Path(index_path).resolve() if index_path else resolved_manifest.with_suffix(".idx")

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    return {
        "manifest": str(resolved_manifest), "num_tasks": len(store),
        "provenance": {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "renderer_version": RENDERER_VERSION,
            "split": split,
            "dataset_seed": MAIN_V1.dataset_seed,
            "sampler_profile": asdict(MAIN_V1),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "bit_generator": np.random.PCG64.__name__,
            "renderer": renderer_provenance(),
            "manifest_sha256": file_sha256(resolved_manifest),
            "index_sha256": file_sha256(resolved_index),
        },
        "counts": {name: dict(values) for name, values in counts.items()},
        "event_directions": {name: dict(values) for name, values in event_directions.items()},
        "unique": {name: len(values) for name, values in hashes.items()},
        "baselines": {
            **baselines,
            "copy_changed_dense_mean_reward": (
                baselines["copy_changed_dense_reward_sum"] / baselines["copy_changed_count"]
                if baselines["copy_changed_count"] else None
            ),
        },
        "failures": failures, "distribution_violations": violations,
        "passed": not failures and not violations,
    }


def audit_split_leakage(manifests: dict[str, str | Path]) -> dict:
    """Check map/current/task identity sets across already-audited split manifests."""
    split_hashes: dict[str, dict[str, set[str]]] = {}
    for split, manifest in manifests.items():
        store = ManifestStore(manifest)
        values = {name: set() for name in ("map", "current", "task")}
        for index in range(len(store)):
            row = store.read(index, verify=False)
            values["map"].add(row["map_hash"])
            values["current"].add(row["current_state_hash"])
            values["task"].add(row["task_hash"])
        split_hashes[split] = values
    intersections: dict[str, dict[str, int]] = {}
    names = sorted(split_hashes)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            intersections[f"{left}__{right}"] = {
                kind: len(split_hashes[left][kind] & split_hashes[right][kind])
                for kind in ("map", "current", "task")
            }
    return {
        "split_sizes": {split: len(values["task"]) for split, values in split_hashes.items()},
        "intersections": intersections,
        "passed": all(count == 0 for pair in intersections.values() for count in pair.values()),
    }
