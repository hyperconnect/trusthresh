import torch

from trusthresh.utils import compute_pr


def maximize_recall_at_given_precision_w_penalty(y_pred: torch.Tensor, y_true: torch.Tensor,
                                                 target_precision: float, alpha: float = 25) -> torch.Tensor:
    """The loss function to solve the following optimization problem.

    maximize_t Recall(t | y_pred, y_true)
    subject to Precision(t | y_pred, y_true) >= target_precision.
    """
    precision, recall = compute_pr(y_pred=y_pred, y_true=y_true)
    return -recall + alpha * torch.clip(target_precision - precision, min=0)


def maximize_precision_at_given_recall_w_penalty(y_pred: torch.Tensor, y_true: torch.Tensor,
                                                 target_recall: float, alpha: float = 25) -> torch.Tensor:
    """The loss function to solve the following optimization problem.

    maximize_t Precision(t | y_pred, y_true)
    subject to Recall(t | y_pred, y_true) >= target_recall.
    """
    precision, recall = compute_pr(y_pred=y_pred, y_true=y_true)
    return -precision + alpha * torch.clip(target_recall - recall, min=0)
