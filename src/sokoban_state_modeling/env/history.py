"""Complete model-visible O0,A0,...,Ot history without hidden environment labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from .core import SokobanObservation


@dataclass(frozen=True)
class AssistantTurn:
    raw_response: str
    action: str


class ObservableTrajectory:
    """Preserve every real frame and assistant turn in chronological order."""

    def __init__(self, instruction: str, initial_observation: SokobanObservation):
        if not instruction:
            raise ValueError("instruction cannot be empty")
        self.instruction = instruction
        self._observations = [initial_observation]
        self._assistant_turns: list[AssistantTurn] = []

    @property
    def observations(self) -> tuple[SokobanObservation, ...]:
        return tuple(self._observations)

    @property
    def assistant_turns(self) -> tuple[AssistantTurn, ...]:
        return tuple(self._assistant_turns)

    def append(self, raw_response: str, action: str, observation: SokobanObservation) -> None:
        """Atomically append Ai and the resulting O(i+1), including no-op frames."""
        if not raw_response:
            raise ValueError("raw assistant response cannot be empty")
        self._assistant_turns.append(AssistantTurn(raw_response=raw_response, action=action))
        self._observations.append(observation)
        self._validate()

    def _validate(self) -> None:
        if len(self._observations) != len(self._assistant_turns) + 1:
            raise RuntimeError("trajectory must alternate O0,A0,...,Ot")

    def trace(self) -> dict[str, Any]:
        self._validate()
        return {
            "actions": [item.action for item in self._assistant_turns],
            "raw_responses": [item.raw_response for item in self._assistant_turns],
            "state_hashes": [item.state_hash for item in self._observations],
            "rgb_hashes": [item.rgb_hash for item in self._observations],
        }

    def model_messages(self) -> list[dict[str, Any]]:
        """Build full history; an adapter may map observation roles to tool roles."""
        self._validate()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.instruction}]}
        ]
        for turn, observation in enumerate(self._observations):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.fromarray(observation.rgb)},
                        {"type": "text", "text": f"Sokoban observation turn={turn}."},
                    ],
                }
            )
            if turn < len(self._assistant_turns):
                messages.append({"role": "assistant", "content": self._assistant_turns[turn].raw_response})
        return messages

