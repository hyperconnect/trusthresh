# TruSThresh
An official PyTorch implementation of ["**Reliable Deicision from Multiple Subtasks through Threshold Optimization: Content Moderation in the Wild**"](https://arxiv.org/abs/2208.07522) (WSDM'23).

## Installation
```bash
pip install -e ./
```

## Quick Example
```python
import numpy as np

from trusthresh.data import Processor, Subtask
from trusthresh.loss import maximize_recall_at_given_precision_w_penalty
from trusthresh.model import TruSThresh, TruSThreshConfig
from trusthresh.operation import And, Col, Or, Sum


# Load data.
x, y = np.load("./dataset/unsmile_pred.npy"), np.load("./dataset/unsmile_gt_usecase_1.npy")

# Write metadata. Here, the order of the metadata must be the same as the column order of x.
metadata = [
	Subtask(name="misogyny", label=["yes", "no"]),
	Subtask(name="religion", label=["yes", "no"]),
	Subtask(name="regionalism", label=["yes", "no"]),
	Subtask(name="ageism", label=["yes", "no"]),
	Subtask(name="sexual_minority", label=["yes", "no"]),
	Subtask(name="male", label=["yes", "no"]),
	Subtask(name="race_and_nationality", label=["yes", "no"]),
	Subtask(name="other_hate", label=["yes", "no"]),
	Subtask(name="swearwords", label=["yes", "no"]),
]

# Write a rule.
rule = Or(
	Col(subtask_name="misogyny", label="yes"),
	Col(subtask_name="religion", label="yes"),
	Col(subtask_name="regionalism", label="yes"),
	Col(subtask_name="ageism", label="yes"),
	Col(subtask_name="sexual_minority", label="yes"),
	Col(subtask_name="male", label="yes"),
	Col(subtask_name="race_and_nationality", label="yes"),
	Col(subtask_name="other_hate", label="yes"),
	Col(subtask_name="swearwords", label="yes"),
)

# Construct a processor,
processor = Processor(
	x=x, y=y, metadata=metadata, rule=rule
)

# Construct a config.
config = TruSThreshConfig(
    initial_threshold=0.5,
    initial_width=0.1,
    tune_width=True,
    device="cpu",
    n_epochs=1000,
    learning_rate=0.01,
    loss_fn=maximize_recall_at_given_precision_w_penalty,
    loss_params={
        "alpha": 32,
        "target_precision": 0.975,
    }
)
solver = TruSThresh(
    processor=processor,
    config=config,
)

result = solver.tune()
```

## Citation
```
@article{
  son2022trusthresh,
  title={Reliable Decision from Multiple Subtasks through Threshold Optimization: Content Moderation in the Wild},
  author={Son, Donghyun and Lew, Byounggyu and Choi, Kwanghee and Baek, Yongsu and Choi, Seungwoo and Shin, Beomjun and Ha, Sungjoo and Chang, Buru},
  journal={arXiv preprint arXiv:2208.07522},
  year={2022},
}
```
