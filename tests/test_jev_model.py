"""CPU contract tests plus optional real-Qwen processor/model integration."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sokoban_state_modeling.jev.backend import QwenBackend
from sokoban_state_modeling.jev.config import Config
from sokoban_state_modeling.jev.dataset import BoardDataset
from sokoban_state_modeling.jev.evaluation import evaluate_dataset
from sokoban_state_modeling.jev.models import BoardInput, CLASSES, probability_vector
from sokoban_state_modeling.jev.prompts import PromptCompiler
from sokoban_state_modeling.jev.scoring import cell_kl, log_probabilities
from sokoban_state_modeling.jev.service import EvaluationService
from sokoban_state_modeling.sft.jev_export import export_jev


class Tokenizer:
    pad_token_id = 0
    chat_template = "test-template"
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, ids):
        return "".join(map(chr, ids))


class Processor:
    tokenizer = Tokenizer()
    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["content"][0]["type"] == "image"
        assert kwargs["enable_thinking"] is False
        return json.dumps(messages) + "Assistant:\n"
    def __call__(self, text, images, **kwargs):
        n = len(text)
        return {"input_ids": torch.tensor([[0, 1, 2]] * n),
                "attention_mask": torch.tensor([[0, 1, 1]] * n),
                "pixel_values": torch.tensor([np.array(im).mean() for im in images]).float()[:, None],
                "image_grid_thw": torch.tensor([[1, 1, 1]] * n)}


class Readout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(128).float()[None, :] / 128)
    def forward(self, pixel_values, logits_to_keep, **kwargs):
        assert logits_to_keep == 1
        return SimpleNamespace(logits=(pixel_values * self.weight)[:, None])


def backend():
    return QwenBackend(Readout(), Processor(), Config(model_path="test", device="cpu", dtype="float32"))


def exported(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"split": "train", "current_state": {
        "grid": ["#####", "#PXO#", "#####"]}, "target": "SECRET"}) + "\n")
    export_jev([path], tmp_path / "data")
    return tmp_path / "data" / "records.jsonl"


def test_probability_key_order_and_validation():
    target = dict.fromkeys(reversed(CLASSES), 0.)
    target["box"] = 1.
    assert probability_vector(target)[3] == 1.
    for bad in ({"box": 1.}, dict.fromkeys(CLASSES, .5), {**target, "wall": float("nan")}):
        with pytest.raises(ValueError):
            probability_vector(bad)


def test_kl_matches_cross_entropy_and_microbatch_gradients():
    torch.manual_seed(4)
    logits = torch.randn(11, 7, requires_grad=True)
    q = torch.nn.functional.one_hot(torch.arange(11) % 7, 7).float()
    assert torch.allclose(cell_kl(logits, q), torch.nn.functional.cross_entropy(logits, q.argmax(-1), reduction="none"))
    soft = torch.rand(11, 7).softmax(-1)
    assert torch.allclose(cell_kl(logits, soft), torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=soft), torch.distributions.Categorical(logits=logits)), atol=1e-6)
    full = (cell_kl(logits[:4], q[:4]).mean() + cell_kl(logits[4:], q[4:]).mean())/2
    expected = torch.autograd.grad(full, logits)[0]
    actual = torch.zeros_like(logits)
    for lo, hi in ((0, 4), (4, 11)):
        for start in range(lo, hi, 3):
            stop = min(start+3, hi)
            loss = cell_kl(logits[start:stop], q[start:stop]).sum()/((hi-lo)*2)
            actual += torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        cell_kl(logits, q*2)


def test_model_image_batch_gradient_and_service_modes():
    model = backend()
    board = BoardInput.create(Path("unused"), 2, 3)
    image = Image.new("RGB", (32, 32), "white")
    both = model.logits(board, board.questions[:2], image)
    single = model.logits(board, board.questions[:1], image)
    torch.testing.assert_close(both[0], single[0])
    assert not torch.equal(single, model.logits(board, board.questions[:1], Image.new("RGB", (32, 32))))
    cell_kl(both, torch.eye(7)[:2]).mean().backward()
    assert model.model.weight.grad is not None
    result = EvaluationService(model, 4).evaluate(board, image)
    assert model.model.training  # evaluation restores previous mode
    assert len(result["answers"]) == 6
    for answer in result["answers"].values():
        assert sum(answer["probabilities"].values()) == pytest.approx(1.)


def test_dataset_alignment_hash_and_no_target_in_prompt(tmp_path):
    path = exported(tmp_path)
    data = BoardDataset(path, "train")
    example = data[0]
    assert example.targets[7][3] == 1.  # row 1 col 2 is box
    text = backend().compiler.render(example.inputs, example.inputs.questions[7])
    assert "SECRET" not in text and "probabilities" not in text
    assert "row=1, column=2" in text
    example.image()
    example.inputs.image_path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        example.image()


def test_evaluation_metrics_and_error_images(tmp_path):
    data = BoardDataset(exported(tmp_path), "train")
    report = evaluate_dataset(EvaluationService(backend(), 8), data, tmp_path)
    assert report["boards"] == 1
    assert sum(c["support"] for c in report["per_class"].values()) == 15
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "error_000.png").exists()


def test_processor_refuses_truncation_and_invalid_padding():
    compiler = PromptCompiler(Processor(), 2)
    board = BoardInput.create(Path("unused"), 1, 1)
    with pytest.raises(ValueError, match="truncation"):
        compiler.batch(board, board.questions, Image.new("RGB", (32, 32)))


def test_config_rejects_unknown_or_bad_values():
    with pytest.raises(TypeError):
        Config.from_dict({"model_path": "test", "typo": 1})
    with pytest.raises(ValueError):
        Config(model_path="test", cell_batch_size=0)
