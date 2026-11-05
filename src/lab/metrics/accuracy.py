import numpy as np
import torch
from transformers import EvalPrediction


def answer_letter_for_metrics(
    logits: torch.Tensor, labels: torch.Tensor, letter_ids: list[int]
) -> torch.Tensor:
    """
    Reduce the [B, L, V] logits to [B, 2]: (predicted option, gold option), options in 0..3.

    The prediction is the letter (A-D) with the highest logit at the position that
    predicts the first answer token.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    letters = torch.tensor(letter_ids, device=logits.device)
    rows = torch.arange(labels.size(0), device=labels.device)
    first = (labels != -100).int().argmax(dim=1)  # first answer token = the letter
    # logits at position t predict token t + 1
    pred = logits[rows, first - 1][:, letters].argmax(dim=-1)
    gold = (labels[rows, first].unsqueeze(1) == letters).int().argmax(dim=1)
    return torch.stack([pred, gold], dim=1)


def compute_accuracy(eval_prediction: EvalPrediction) -> dict[str, float]:
    preds = np.asarray(eval_prediction.predictions).reshape(-1, 2)
    return {'accuracy': float((preds[:, 0] == preds[:, 1]).mean())}
