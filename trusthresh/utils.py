from typing import List
from typing import Tuple

import torch

from trusthresh.operation import Op


def compute_pr(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-10) -> Tuple[float]:
    TP = torch.sum(y_true * y_pred)
    FP = torch.sum((1 - y_true) * y_pred)
    TN = torch.sum((1 - y_true) * (1 - y_pred))
    FN = torch.sum(y_true * (1 - y_pred))
    precision, recall = (TP + eps) / (TP + FP + eps), (TP + eps) / (TP + FN + eps)
    return precision, recall


def compute_pr_from_rule(x: torch.Tensor, y: torch.Tensor,
                         rule: Op, thresholds: List[float]) -> Tuple[float]:
    x = x - torch.Tensor(thresholds)
    x = (x > 0).type(torch.float)
    pred = rule(x)
    precision, recall = compute_pr(pred, y)
    return precision, recall
