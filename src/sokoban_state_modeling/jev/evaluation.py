"""Raw board accuracy, per-class metrics and bounded visual error reports."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from .dataset import BoardDataset
from .models import CLASSES, SYMBOLS
from .service import EvaluationService


def evaluate_dataset(service: EvaluationService, dataset: BoardDataset,
                     output: Path | None = None, max_errors: int = 20) -> dict:
    confusion = np.zeros((7, 7), dtype=np.int64)
    exact, legal, errors = 0, 0, []
    groups: dict[str, dict] = {}
    entity_exact = dict(player=0, box=0, goal=0)
    for index in range(len(dataset)):
        example = dataset[index]
        image = example.image()
        response = service.evaluate(example.inputs, image)
        truth = np.array([max(range(7), key=q.__getitem__) for q in example.targets])
        predicted = np.array([CLASSES.index(response["answers"][q.key]["choice"])
                              for q in example.inputs.questions])
        np.add.at(confusion, (truth, predicted), 1)
        is_exact = bool(np.array_equal(truth, predicted))
        exact += is_exact
        legal += int(np.isin(predicted, [5, 6]).sum() == 1 and
                     np.isin(predicted, [3, 4]).sum() == np.isin(predicted, [2, 4, 6]).sum())
        for name, classes in (("player", [5, 6]), ("box", [3, 4]), ("goal", [2, 4, 6])):
            entity_exact[name] += int(np.array_equal(np.isin(truth, classes), np.isin(predicted, classes)))
        size = f"{example.inputs.height}x{example.inputs.width}"
        group = groups.setdefault(size, {"boards": 0, "exact": 0, "cells": 0, "correct_cells": 0})
        group["boards"] += 1
        group["exact"] += int(is_exact)
        group["cells"] += len(truth)
        group["correct_cells"] += int((truth == predicted).sum())
        if not is_exact and len(errors) < max_errors:
            error = {"sample_id": example.sample_id, "image": str(example.inputs.image_path),
                     "predicted_grid": response["grid"], "wrong_cells": []}
            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            for i in np.flatnonzero(truth != predicted):
                q = example.inputs.questions[int(i)]
                error["wrong_cells"].append({"row": q.row, "column": q.column,
                                            "target": CLASSES[truth[i]], "prediction": CLASSES[predicted[i]]})
                x, y = preview.width / example.inputs.width, preview.height / example.inputs.height
                draw.rectangle((q.column*x, q.row*y, (q.column+1)*x-1, (q.row+1)*y-1), outline="red", width=2)
            if output:
                preview.save(output / f"error_{len(errors):03d}.png")
            errors.append(error)
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp, support, selected = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        precision, recall = tp / selected if selected else 0., tp / support if support else 0.
        per_class[name] = {"support": support, "precision": precision, "recall": recall,
                           "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.}
    for group in groups.values():
        group["board_exact"] = group["exact"] / group["boards"]
        group["cell_accuracy"] = group["correct_cells"] / group["cells"]
    report = {"split": dataset.split, "boards": len(dataset), "board_exact": exact / len(dataset),
              "cell_accuracy": float(np.trace(confusion) / confusion.sum()),
              "structural_validity": legal / len(dataset), "per_class": per_class,
              "entity_exact": {k: v / len(dataset) for k, v in entity_exact.items()},
              "confusion_matrix": confusion.tolist(), "by_size": groups, "errors": errors}
    if output:
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
