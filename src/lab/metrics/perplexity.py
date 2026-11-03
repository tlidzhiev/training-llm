import math

import torch
from transformers import EvalPrediction


def token_loss_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    loss_sum = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction='sum',
    )
    num_tokens = (shift_labels != -100).sum()
    return torch.stack([loss_sum, num_tokens.to(loss_sum.dtype)]).unsqueeze(0)


def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    reductions = torch.as_tensor(eval_prediction.predictions).reshape(-1, 2)
    loss_sum, num_tokens = reductions.sum(dim=0).tolist()
    if num_tokens == 0:
        return {}
    return {'perplexity': math.exp(loss_sum / num_tokens)}
