"""Materialise deterministic state/action manifests with resumable chunks."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sokoban_state_modeling import GENERATOR_VERSION, RENDERER_VERSION, SCHEMA_VERSION
from sokoban_state_modeling.engine import generate_room_snapshots
from sokoban_state_modeling.state.codec import (StateSnapshot, map_hash, room_to_state, state_hash,
                                                task_hash, terrain_family_hash)
from sokoban_state_modeling.state.transition import DELTAS, is_terminal
from sokoban_state_modeling.state.validation import solve_forward, validation_errors
from sokoban_state_modeling.utils.seeding import set_seed
from .action_sampler import ActionSampler
from .manifest_store import build_index
from .sampler import DISTANCE_RANGES, MAIN_V1, QuotaSchedule, SamplerProfile, reverse_bucket

SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
# Disjoint map ownership weighted approximately by sqrt(formal split size). This
# minimizes total rejection work without starving the smaller validation split.
SPLIT_INTERVALS = {"train": (0.0, 0.61), "validation": (0.61, 0.73), "test": (0.73, 1.0)}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _belongs_to_split(digest: str, split: str) -> bool:
    value = int.from_bytes(bytes.fromhex(digest)[:8], "big") / 2**64
    low, high = SPLIT_INTERVALS[split]
    return low <= value < high


def _reachable_and_distance(state: StateSnapshot) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    player = tuple(map(int, np.argwhere(state.entity == 1)[0]))
    free = (state.fixed != 0) & (state.entity != 2)
    reachable = {player}
    queue = deque([player])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DELTAS.values():
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < state.shape[0] and 0 <= nxt[1] < state.shape[1] and free[nxt] and nxt not in reachable:
                reachable.add(nxt)
                queue.append(nxt)
    boxes = [tuple(map(int, value)) for value in np.argwhere(state.entity == 2)]
    contacts = {
        (br - dr, bc - dc)
        for br, bc in boxes for dr, dc in DELTAS.values()
        if (br - dr, bc - dc) in reachable
    }
    distances: dict[tuple[int, int], int] = {}
    queue = deque()
    for contact in contacts:
        distances[contact] = 0
        queue.append(contact)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DELTAS.values():
            nxt = (r + dr, c + dc)
            if nxt in reachable and nxt not in distances:
                distances[nxt] = distances[(r, c)] + 1
                queue.append(nxt)
    return reachable, distances


def _adjacent_type(state: StateSnapshot, position: tuple[int, int]) -> str | None:
    r, c = position
    for dr, dc in DELTAS.values():
        box = (r + dr, c + dc)
        beyond = (r + 2 * dr, c + 2 * dc)
        if 0 <= box[0] < state.shape[0] and 0 <= box[1] < state.shape[1] and state.entity[box] == 2:
            pushable = (
                0 <= beyond[0] < state.shape[0] and 0 <= beyond[1] < state.shape[1]
                and state.fixed[beyond] != 0 and state.entity[beyond] == 0
            )
            if pushable:
                return "pushable_adjacent"
    return "blocked_adjacent"


def place_player(
    state: StateSnapshot,
    bucket: str,
    adjacent_type: str | None,
    require_target: bool,
    rng: np.random.Generator,
) -> tuple[StateSnapshot, int] | None:
    reachable, distances = _reachable_and_distance(state)
    low, high = DISTANCE_RANGES[bucket]
    candidates = []
    for position in sorted(reachable):
        distance = distances.get(position)
        if distance is None or distance < low or (high is not None and distance > high):
            continue
        if require_target and state.fixed[position] != 2:
            continue
        if bucket == "adjacent" and adjacent_type and _adjacent_type(state, position) != adjacent_type:
            continue
        candidates.append((position, distance))
    if not candidates:
        return None
    position, distance = candidates[int(rng.integers(len(candidates)))]
    entity = state.entity.copy()
    entity[entity == 1] = 0
    entity[position] = 1
    return StateSnapshot(state.fixed, entity), distance


def _wall_stats(state: StateSnapshot) -> tuple[float, float]:
    interior = state.fixed[1:-1, 1:-1]
    wall_ratio = float(np.mean(interior == 0))
    non_walls = np.argwhere(interior != 0)
    corridor = 0
    for ir, ic in non_walls:
        r, c = int(ir + 1), int(ic + 1)
        free = [state.fixed[r + dr, c + dc] != 0 for dr, dc in DELTAS.values()]
        corridor += int((free[0] and free[1] and not free[2] and not free[3]) or (free[2] and free[3] and not free[0] and not free[1]))
    return wall_ratio, corridor / len(non_walls) if len(non_walls) else 0.0


def _box_target_distance(state: StateSnapshot) -> int:
    boxes = np.argwhere(state.entity == 2)
    targets = np.argwhere(state.fixed == 2)
    return int(min(np.abs(box - target).sum() for box in boxes for target in targets))


def _static_attributes(state: StateSnapshot, wall_ratio: float, corridor_ratio: float) -> dict[str, bool]:
    _, distances = _reachable_and_distance(state)
    player = tuple(map(int, np.argwhere(state.entity == 1)[0]))
    boxes = state.entity == 2
    boxes_on_target = int(np.sum(boxes & (state.fixed == 2)))
    return {
        "is_player_adjacent_box": distances.get(player) == 0,
        # Placeholder keeps the historical JSON key order. The solver-derived
        # value replaces it later without moving the key.
        "is_near_completion": False,
        "has_box_on_target": boxes_on_target > 0,
        "is_player_on_target": state.fixed[player] == 2,
        "is_narrow_or_high_wall": wall_ratio >= 0.35 or corridor_ratio >= 0.50,
    }


def _category_can_match_without_solver(category: str, attrs: dict[str, bool]) -> bool:
    """Reject category mismatches whose predicates do not need shortest path."""
    if category == "ordinary_solvable":
        return not any(attrs.values())
    if category == "player_near_box":
        return attrs["is_player_adjacent_box"]
    if category == "near_completion":
        return True
    if category == "box_on_target":
        return attrs["has_box_on_target"]
    if category == "player_on_target":
        return attrs["is_player_on_target"]
    if category == "narrow_or_high_wall":
        return attrs["is_narrow_or_high_wall"]
    raise ValueError(category)


def _category_matches(category: str, state: StateSnapshot, shortest: int, attrs: dict[str, bool]) -> bool:
    if category == "ordinary_solvable":
        return not any(attrs.values())
    if category == "player_near_box":
        return attrs["is_player_adjacent_box"]
    if category == "near_completion":
        return attrs["is_near_completion"]
    if category == "box_on_target":
        count = int(np.sum((state.entity == 2) & (state.fixed == 2)))
        return 0 < count < int(np.sum(state.entity == 2))
    if category == "player_on_target":
        return attrs["is_player_on_target"]
    if category == "narrow_or_high_wall":
        return attrs["is_narrow_or_high_wall"]
    raise ValueError(category)


class DatasetBuilder:
    def __init__(
        self, output_dir: str | Path, split: str, total: int,
        profile: SamplerProfile = MAIN_V1, chunk_size: int = 100,
    ):
        if split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {tuple(SPLIT_IDS)}")
        self.output_dir = Path(output_dir).resolve()
        self.split = split
        self.total = int(total)
        self.chunk_size = int(chunk_size)
        if self.total <= 0 or self.chunk_size <= 0:
            raise ValueError("total and chunk_size must be positive")
        self.profile = profile
        self.schedule = QuotaSchedule(
            total,
            profile,
            profile.dataset_seed + SPLIT_IDS[split] * 10_000_000,
            split=split,
        )
        self.action_sampler = ActionSampler()
        self.used_maps: set[str] = set()
        self.used_states: set[str] = set()
        self.used_tasks: set[str] = set()

    def _candidate(self, index: int, attempt: int, assignment: dict[str, Any]):
        generation_seed = self.profile.dataset_seed + SPLIT_IDS[self.split] * 10_000_000 + index
        seed_sequence = np.random.SeedSequence([generation_seed, attempt])
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        try:
            with set_seed(int(seed_sequence.generate_state(1, dtype=np.uint32)[0])):
                _, snapshots, d_max = generate_room_snapshots(
                    dim=tuple(assignment["dim_room"]), topology_steps=assignment["topology_steps"],
                    p_change_directions=assignment["p_change_directions"], num_boxes=assignment["num_boxes"],
                    reverse_search_depth=assignment["reverse_search_depth"], max_reverse_nodes=self.profile.max_reverse_nodes,
                    max_per_depth=self.profile.max_snapshot_candidates_per_depth, rng=rng,
                )
        except (RuntimeError, RuntimeWarning, IndexError):
            return None
        snapshots = [item for item in snapshots if reverse_bucket(item.reverse_depth, d_max) == assignment["reverse_progress"]]
        rng.shuffle(snapshots)
        for snapshot in snapshots[: self.profile.max_snapshot_candidates_per_bucket]:
            state = room_to_state(snapshot.room_fixed, snapshot.room_state)
            placed = place_player(
                state, assignment["player_distance"], assignment["adjacent_type"],
                assignment["primary_state_category"] == "player_on_target", rng,
            )
            if placed is None:
                continue
            state, player_distance = placed
            errors = validation_errors(state, assignment["num_boxes"])
            if errors or is_terminal(state):
                continue
            mh, sh = map_hash(state), state_hash(state)
            if (not _belongs_to_split(terrain_family_hash(state), self.split)
                    or mh in self.used_maps or sh in self.used_states):
                continue
            wall_ratio, corridor_ratio = _wall_stats(state)
            if wall_ratio > 0.55:
                continue
            static_attrs = _static_attributes(state, wall_ratio, corridor_ratio)
            if not _category_can_match_without_solver(
                assignment["primary_state_category"], static_attrs
            ):
                continue
            solver = solve_forward(state, self.profile.max_forward_bfs_nodes)
            if solver.status != "solvable" or solver.shortest_steps is None:
                continue
            attrs = dict(static_attrs)
            attrs["is_near_completion"] = 1 <= solver.shortest_steps <= 3
            if not _category_matches(assignment["primary_state_category"], state, solver.shortest_steps, attrs):
                continue
            required_events = None
            category = assignment["primary_state_category"]
            if category == "box_on_target":
                required_events = {"box_leaves_target"}
            elif category == "player_on_target":
                required_events = {"player_target_transition"}
            elif category == "player_near_box":
                required_events = (
                    {"blocked_box_noop"}
                    if assignment["adjacent_type"] == "blocked_adjacent"
                    else {"box_push", "box_enters_target", "box_leaves_target"}
                )
            elif assignment["terminal"]:
                required_events = {"box_enters_target"}
            block = self.action_sampler.sample(
                state, assignment["action_length"], assignment["terminal"], rng,
                required_events=required_events,
            )
            if block is None:
                continue
            th = task_hash(state, block.actions)
            if th in self.used_tasks:
                continue
            return generation_seed, snapshot, d_max, state, player_distance, solver, wall_ratio, corridor_ratio, attrs, block, mh, sh, th
        return None

    def _make_row(self, index: int) -> dict[str, Any]:
        assignment = self.schedule.assignment(index)
        candidate = None
        for attempt in range(self.profile.max_task_attempts):
            candidate = self._candidate(index, attempt, assignment)
            if candidate is not None:
                break
        if candidate is None:
            raise RuntimeError(
                f"failed task {index} after {self.profile.max_task_attempts} attempts: {assignment}"
            )
        generation_seed, snapshot, d_max, state, player_distance, solver, wall_ratio, corridor_ratio, attrs, block, mh, sh, th = candidate
        return {
            "schema_version": SCHEMA_VERSION, "task_id": f"{self.split}_{index + 1:08d}",
            "task_index": index, "split": self.split, "generator_version": GENERATOR_VERSION,
            "renderer_version": RENDERER_VERSION, "generation_seed": generation_seed,
            "accepted_attempt": attempt, "sampler_profile": self.profile.name,
            "resolved_generator_config": {
                "dim_room": list(assignment["dim_room"]), "num_boxes": assignment["num_boxes"],
                "topology_steps": assignment["topology_steps"], "topology_multiplier": assignment["topology_multiplier"],
                "p_change_directions": assignment["p_change_directions"],
                "reverse_search_depth": assignment["reverse_search_depth"], "reverse_progress_bucket": assignment["reverse_progress"],
                "sampled_reverse_depth": snapshot.reverse_depth, "reverse_d_max": d_max,
                "player_placement_mode": "distance_bucket", "player_placement_bucket": assignment["player_distance"],
                "player_move_probability": 0.0, "player_continue_probability": 0.0, "player_reposition_steps": 0,
            },
            "map_hash": mh, "current_state_hash": sh, "next_state_hash": state_hash(block.next_state),
            "query_actions": block.actions, "task_hash": th,
            "current_state": state.to_dict(), "next_state": block.next_state.to_dict(),
            "event_sequence": block.events, "event_subtype_sequence": block.event_subtypes,
            "executed_action_count": len(block.actions), "terminated": block.terminated,
            "metadata": {
                "primary_state_category": assignment["primary_state_category"], "solver_status": solver.status,
                "shortest_solution_steps": solver.shortest_steps, "wall_ratio": wall_ratio,
                "corridor_cell_ratio": corridor_ratio, "player_box_distance": player_distance,
                "player_box_distance_bucket": assignment["player_distance"],
                "adjacent_type": assignment["adjacent_type"], "box_target_distance": _box_target_distance(state),
                "query_length": len(block.actions), "state_attributes": attrs,
            },
        }

    def _remember_row(
        self, row: dict[str, Any], expected_index: int, *, restore_sampler: bool,
    ) -> None:
        if row.get("task_index") != expected_index or row.get("split") != self.split:
            raise ValueError(f"checkpoint row {expected_index} has wrong identity")
        if row.get("schema_version") != SCHEMA_VERSION or row.get("generator_version") != GENERATOR_VERSION:
            raise ValueError(f"checkpoint row {expected_index} has incompatible version")
        expected_seed = self.profile.dataset_seed + SPLIT_IDS[self.split] * 10_000_000 + expected_index
        if row.get("generation_seed") != expected_seed or row.get("sampler_profile") != self.profile.name:
            raise ValueError(f"checkpoint row {expected_index} belongs to another sampler run")
        for key, seen in (
            ("map_hash", self.used_maps), ("current_state_hash", self.used_states),
            ("task_hash", self.used_tasks),
        ):
            value = row[key]
            if value in seen:
                raise ValueError(f"duplicate {key} in generation checkpoint")
            seen.add(value)
        if restore_sampler:
            for event, action in zip(row["event_sequence"], row["query_actions"]):
                self.action_sampler.event_counts[event] += 1
                self.action_sampler.direction_counts[event][action] += 1

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "renderer_version": RENDERER_VERSION,
            "split": self.split,
            "total": self.total,
            # Canonical JSON types make the in-memory payload compare equal to the
            # payload read back during a later --resume process.
            "profile": json.loads(json.dumps(asdict(self.profile), default=_json_default)),
            "chunk_size": self.chunk_size,
        }

    def build(self, resume: bool = False) -> tuple[Path, Path]:
        manifest_dir = self.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{self.split}.jsonl"
        index_path = manifest_path.with_suffix(".idx")
        if resume and manifest_path.exists() and not index_path.exists():
            build_index(manifest_path, index_path)
        if resume and manifest_path.exists() and index_path.exists():
            return manifest_path, index_path
        if manifest_path.exists() or index_path.exists():
            raise FileExistsError(
                f"refusing to overwrite an existing dataset: {manifest_path}"
            )
        chunks_dir = self.output_dir / "generation_chunks" / self.split
        checkpoint_path = self.output_dir / f".{self.split}.generation.json"
        expected_checkpoint = self._checkpoint_payload()
        existing_chunks = sorted(chunks_dir.glob("chunk_*.jsonl")) if chunks_dir.exists() else []
        if resume:
            if checkpoint_path.exists():
                actual = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if actual != expected_checkpoint:
                    raise ValueError("generation checkpoint does not match split/size/profile/versions")
            elif existing_chunks:
                raise ValueError("generation chunks exist without their checkpoint metadata")
        elif checkpoint_path.exists() or existing_chunks:
            raise FileExistsError("unfinished generation exists; rerun with --resume")

        chunks_dir.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.exists():
            checkpoint_partial = checkpoint_path.with_suffix(".json.partial")
            checkpoint_partial.write_text(
                json.dumps(expected_checkpoint, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            checkpoint_partial.replace(checkpoint_path)

        next_index = 0
        for chunk_path in existing_chunks:
            rows = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines() if line]
            if not rows or (len(rows) != self.chunk_size and next_index + len(rows) != self.total):
                raise ValueError(f"invalid committed generation chunk {chunk_path}")
            for row in rows:
                self._remember_row(row, next_index, restore_sampler=True)
                next_index += 1
        if next_index > self.total:
            raise ValueError("generation checkpoint contains more rows than requested")

        chunk_rows: list[dict[str, Any]] = []
        for index in range(next_index, self.total):
            row = self._make_row(index)
            # ActionSampler.sample already committed this row's event counters.
            self._remember_row(row, index, restore_sampler=False)
            chunk_rows.append(row)
            if len(chunk_rows) == self.chunk_size or index + 1 == self.total:
                first = int(chunk_rows[0]["task_index"])
                last = int(chunk_rows[-1]["task_index"])
                chunk_path = chunks_dir / f"chunk_{first:08d}_{last:08d}.jsonl"
                partial_chunk = chunk_path.with_suffix(".jsonl.partial")
                with partial_chunk.open("w", encoding="utf-8", newline="\n") as output:
                    for item in chunk_rows:
                        output.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                partial_chunk.replace(chunk_path)
                chunk_rows.clear()
                print(f"[{self.split}] generated {index + 1}/{self.total}", flush=True)

        partial_manifest_path = manifest_path.with_suffix(".jsonl.partial")
        with partial_manifest_path.open("wb") as manifest:
            for chunk_path in sorted(chunks_dir.glob("chunk_*.jsonl")):
                manifest.write(chunk_path.read_bytes())
            manifest.flush()
            os.fsync(manifest.fileno())
        partial_manifest_path.replace(manifest_path)
        build_index(manifest_path, index_path)
        for chunk_path in chunks_dir.glob("chunk_*.jsonl"):
            chunk_path.unlink()
        chunks_dir.rmdir()
        chunks_dir.parent.rmdir()
        checkpoint_path.unlink()
        return manifest_path, index_path
