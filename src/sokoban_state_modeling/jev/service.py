"""Local cell batching; does not implement HTTP or shared-prefix caching."""
from __future__ import annotations
import torch
from PIL import Image
from .backend import QwenBackend
from .models import BoardInput, CLASSES, SYMBOLS
from .scoring import log_probabilities


class EvaluationService:
    def __init__(self, backend: QwenBackend, cell_batch_size: int):
        if cell_batch_size < 1:
            raise ValueError("cell_batch_size must be positive")
        self.backend, self.cell_batch_size = backend, cell_batch_size

    def evaluate(self, board: BoardInput, image: Image.Image | None = None) -> dict:
        if image is None:
            with Image.open(board.image_path) as source:
                image = source.convert("RGB")
        was_training = self.backend.model.training
        self.backend.model.eval()
        probabilities = []
        try:
            with torch.inference_mode():
                for start in range(0, len(board.questions), self.cell_batch_size):
                    questions = board.questions[start:start + self.cell_batch_size]
                    probabilities.extend(log_probabilities(self.backend.logits(board, questions, image)).exp().cpu().tolist())
        finally:
            self.backend.model.train(was_training)
        answers, grid = {}, [[""] * board.width for _ in range(board.height)]
        for question, probs in zip(board.questions, probabilities, strict=True):
            index = max(range(7), key=probs.__getitem__)
            answers[question.key] = {"type": "choice", "choice": CLASSES[index],
                                     "probabilities": dict(zip(CLASSES, probs))}
            grid[question.row][question.column] = SYMBOLS[index]
        return {"height": board.height, "width": board.width, "answers": answers,
                "grid": ["".join(row) for row in grid]}
