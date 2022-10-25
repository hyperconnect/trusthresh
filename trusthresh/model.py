# This codes are borrowed and modified from https://github.com/topel/sglthresh.

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from trusthresh.data import Processor
from trusthresh.utils import compute_pr_from_rule


class TruSThreshSurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input, width)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        input, width = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        grad = grad_input * 0.5 * torch.cos(input / width * math.pi / 2) * math.pi / 2 / width
        grad_width = -grad_input * 0.25 * math.pi * input * torch.cos(math.pi / 2 * input / width) / width**2
        
        mask = torch.logical_or(input < -width, input > width)
        grad[mask] = 0
        grad_width[mask] = 0
        return grad, grad_width


class TruSThreshModel(nn.Module):
    def __init__(
        self,
        n_threshold: int,
        initial_threshold: float = 0.5,
        initial_width: float = 0.01,
        tune_width: bool = True,
        device: Any = "cpu"
    ) -> None:
        super().__init__()
        self.threshold_fn = TruSThreshSurrogateHeaviside.apply
        self.thresholds = nn.Parameter(initial_threshold * torch.ones(n_threshold), requires_grad=True)

        # Sigmoid trick to lay the width between 0 and 1.
        w = torch.tensor(initial_width, dtype=torch.float32)
        self.width = nn.Parameter(torch.log(w / (1 - w)) * torch.ones(n_threshold), requires_grad=tune_width)

        self.device = device
        self.thresholds = self.thresholds.to(device)
        self.width = self.width.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.threshold_fn(x - self.thresholds, torch.sigmoid(self.width))

    def clamp(self) -> None:
        self.thresholds.data.clamp_(min=0., max=1.)


@dataclass
class TruSThreshConfig:
    initial_threshold: float
    initial_width: float
    tune_width: bool
    device: Any
    n_epochs: int
    learning_rate: float
    loss_fn: Callable[[Any], torch.Tensor]
    loss_params: Dict[str, Any]


class TruSThresh:
    def __init__(
        self,
        processor: Processor,
        config: TruSThreshConfig,
    ) -> None:
        self.loss_fn = partial(config.loss_fn, **config.loss_params)
        self.processor = processor

        self.model = TruSThreshModel(
            n_threshold=len(self.processor.numeric_ops),
            initial_threshold=config.initial_threshold,
            initial_width=config.initial_width,
            tune_width=config.tune_width,
            device=config.device,
        )

        self.n_epochs = config.n_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.rank_score = self._build_rank_score()

    def _build_rank_score(self) -> pd.DataFrame:
        rank_score = pd.DataFrame(self.processor.x).rank() / len(self.processor.x)
        return rank_score

    def _covert_threshold_from_rank_to_prob(self, rank_thresholds: List[float]) -> List[float]:
        prob_thresholds: List[float] = []
        for i, rank_threshold in enumerate(rank_thresholds):
            probs = sorted(self.processor.x[:, i])
            ranks = sorted(self.rank_score.iloc[:, i])

            rank_idx = bisect_right(ranks, rank_threshold)
            if rank_idx == len(ranks):
                prob_threshold = (probs[rank_idx - 1] + 1) * 0.5
            elif rank_idx > 0:
                prob_threshold = (probs[rank_idx - 1] + probs[rank_idx]) * 0.5
            else:
                prob_threshold = probs[rank_idx] * 0.5
            prob_thresholds.append(prob_threshold)
        return prob_thresholds

    def tune(self, verbose: bool = False, verbose_interval: int = 100) \
        -> Dict[str, Any]:
        score = self.rank_score.to_numpy()

        inputs = torch.Tensor(score).to(self.model.device)
        targets = torch.Tensor(self.processor.y).to(self.model.device)

        best_loss = 1e10
        best_model_state_dict = deepcopy(self.model).state_dict()

        self.model.train()
        for epoch in range(self.n_epochs):
            outputs = self.model(inputs)

            loss = self.loss_fn(self.processor.rule(outputs), targets)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state_dict = deepcopy(self.model).state_dict()
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.clamp()

            if verbose and (epoch + 1) % verbose_interval == 0:
                print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, self.n_epochs, loss.item()))

        self.model.load_state_dict(best_model_state_dict)
        learned_thresholds = self.model.thresholds.clone().detach().tolist()
        learned_thresholds = self._covert_threshold_from_rank_to_prob(learned_thresholds)
        
        precision, recall = compute_pr_from_rule(torch.Tensor(self.processor.x).to(self.model.device),
                                                 targets, self.processor.rule, learned_thresholds)
        
        print("Result: Precision {:.4f}, Recall: {:.4f}".format(precision ,recall))
        return {
            "precision": precision,
            "recall": recall,
            "thresholds": learned_thresholds,            
        }
