from __future__ import annotations

import json
from typing import Iterable


def system_prompt() -> str:
    return "You are a precise Sokoban visual state-transition model."


def build_task_prompt(rows: int, cols: int, actions: Iterable[str], image_placeholder: str = "<image>") -> str:
    actions_json = json.dumps(list(actions), ensure_ascii=True, separators=(",", ":"))
    return f"""{image_placeholder}
You are predicting Sokoban state transitions from an RGB image.

The image shows the current {rows}x{cols} board. Use exactly one character per cell:
# = wall
_ = empty floor
O = empty target (uppercase letter O)
X = box on floor
P = player on floor
* = box on target
+ = player on target

The environment executes these actions in order: {actions_json}
Action semantics:
- Up/Down/Left/Right attempts to move the player by one cell.
- Moving into a wall has no effect.
- Moving into a box pushes it by one cell only if the cell behind it is free floor or a target; otherwise the action has no effect.
- Every listed action is executed. If the episode terminates, it can only happen on the final listed action.

Infer the current symbolic board from the image, then predict the single final board after all actions.
Return no explanation. Do not output the literal placeholder text "...grid rows...". Output exactly {rows} grid rows between each pair of tags, with exactly {cols} symbols per row. Use this exact tag order:
<perception>
...grid rows...
</perception>
<prediction>
...grid rows...
</prediction>"""
