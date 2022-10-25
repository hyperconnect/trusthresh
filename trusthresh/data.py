from dataclasses import dataclass
from typing import Dict
from typing import List

import numpy as np
import numpy.typing as npt

from trusthresh.operation import Col
from trusthresh.operation import LogicOp
from trusthresh.operation import NumericOp
from trusthresh.operation import Op
from trusthresh.operation import Sum


@dataclass
class Subtask:
    name: str
    label: List[str]

    def get_indices_from_start_idx(self, start_idx: int) -> Dict[str, int]:
        size = len(self.label)
        return {_label: _idx for _label, _idx in zip(self.label, range(start_idx, start_idx + size))}


class Processor:
    def __init__(self, x: npt.NDArray[np.float_], y: npt.NDArray[np.bool_],
                 metadata: List[Subtask], rule: Op) -> None:

        self.metadata = metadata
        self.metadata_indices = self._construct_indices_from_metadata()

        self.rule = rule
        self.numeric_ops = self._parse_numeric_ops_from_rule()
        
        self.x = self._preprocess_x_from_numeric_ops(x)
        self.y = y.astype(int)

    def _preprocess_x_from_numeric_ops(self, x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        idx = 0
        preprocessed_x = []
        for op in self.numeric_ops:
            if isinstance(op, Col):
                _x = x[:, self.metadata_indices[op.subtask_name][op.label]]
            elif isinstance(op, Sum):
                _indices = [self.metadata_indices[op.subtask_name][label]
                for label in op.labels]
                _x = np.sum(x[:, _indices], axis=-1)
            preprocessed_x.append(_x)

            op.update_idx(idx)
            idx += 1
        return np.stack(preprocessed_x, axis=-1)

    def _parse_numeric_ops_from_rule(self) -> List[NumericOp]:
        def _parse_numeric_ops_from_rule_helper(op: Op) -> List[NumericOp]:
            if isinstance(op, LogicOp):
                res: List[NumericOp] = []
                for _op in op.ops:
                    res += _parse_numeric_ops_from_rule_helper(_op)
                return res
            else:
                return [op]
        return _parse_numeric_ops_from_rule_helper(self.rule)

    def _construct_indices_from_metadata(self) -> Dict[str, Dict[str, int]]:
        indices: Dict[str, Dict[str, int]] = dict()
        idx = 0
        for subtask in self.metadata:
            indices[subtask.name] = subtask.get_indices_from_start_idx(idx)
            idx += len(subtask.label)
        return indices
