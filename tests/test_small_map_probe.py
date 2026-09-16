from PIL import Image

from sokoban_state_modeling.scripts.test_qwen35_small_map import (
    CURRENT_GRID,
    NEXT_GRID,
    _report,
    build_messages,
)


def test_probe_uses_multimodal_training_prompt_and_accepts_exact_answer():
    messages = build_messages(Image.new("RGB", (80, 80)))
    assert messages[1]["content"][0]["url"].startswith("data:image/png;base64,")
    assert "current 5x5 board" in messages[1]["content"][1]["text"]
    assert '["UP"]' in messages[1]["content"][1]["text"]

    answer = (
        "<perception>\n"
        + "\n".join(CURRENT_GRID)
        + "\n</perception>\n<prediction>\n"
        + "\n".join(NEXT_GRID)
        + "\n</prediction>"
    )
    report = _report(answer)
    assert report["format_exact"] is True
    assert report["perception_exact"] is True
    assert report["prediction_exact"] is True
    assert report["joint_exact"] is True
    assert report["exact_only_reward"] == 1.0
    assert report["dense_reward"] == 1.0
