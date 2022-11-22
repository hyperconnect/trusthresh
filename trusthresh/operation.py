from abc import ABC
from abc import abstractmethod
from functools import reduce
from typing import List

import torch


class Op(ABC):
    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class NumericOp(Op):
    idx: int

    def update_idx(self, idx: int) -> None:
        self.idx = idx

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.select(1, self.idx)


class LogicOp(Op):
    def __init__(self, *ops: Op) -> None:
        self.ops = ops


class Col(NumericOp):
    def __init__(self, subtask_name: str, label: str):
        self.subtask_name = subtask_name
        self.label = label


class Sum(NumericOp):
    def __init__(self, subtask_name: str, labels: List[str]):
        self.subtask_name = subtask_name
        self.labels = labels


class And(LogicOp):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        subs = [op(inputs) for op in self.ops]
        return reduce(torch.multiply, subs)


class Or(LogicOp):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        subs = [op(inputs) for op in self.ops]
        return 1 - reduce(torch.multiply, [1-res for res in subs],
                          torch.tensor(1.0).to(inputs.device))
