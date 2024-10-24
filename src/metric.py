from typing import Callable, Dict, Iterable, Sequence, Union

import torch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader


class MeanAccumulator:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: Union[float, torch.Tensor]) -> None:
        if isinstance(value, torch.Tensor):
            self.sum += float(value.sum())
            self.count += value.numel()
        else:
            self.sum += value
            self.count += 1

    def compute(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

class MaxAccumulator:
    def __init__(self):
        self.max_value = float('-inf')

    def update(self, value: Union[float, torch.Tensor]) -> None:
        if isinstance(value, torch.Tensor):
            self.max_value = max(self.max_value, float(value.max()))
        else:
            self.max_value = max(self.max_value, value)

    def compute(self) -> float:
        return self.max_value

class MetricsManager:
    def __init__(self):
        self.mean_pointwise_relative_error = MeanAccumulator()
        self.max_pointwise_relative_error = MaxAccumulator()
        self.mean_batch_peak_relative_error = MeanAccumulator()
        self.max_batch_peak_relative_error = MaxAccumulator()
        self.loss = MeanAccumulator()

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        rel_errors = calculate_relative_errors(preds, targets)
        batch_max_value_rel_error = float(
            torch.abs(preds.abs().max() - targets.abs().max()) / targets.abs().max())

        self.mean_pointwise_relative_error.update(rel_errors)
        self.max_pointwise_relative_error.update(rel_errors)
        self.mean_batch_peak_relative_error.update(batch_max_value_rel_error)
        self.max_batch_peak_relative_error.update(batch_max_value_rel_error)
        self.loss.update(loss)

    def compute(self) -> Dict[str, float]:
        return {
            'loss': self.loss.compute(),
            'mean_pointwise_relative_error': self.mean_pointwise_relative_error.compute(),
            'max_pointwise_relative_error': self.max_pointwise_relative_error.compute(),
            'mean_batch_peak_relative_error': self.mean_batch_peak_relative_error.compute(),
            'max_batch_peak_relative_error': self.max_batch_peak_relative_error.compute()
        }

def calculate_relative_errors(preds: torch.Tensor, targets: torch.Tensor, zero_tol: float = 1e-5) -> torch.Tensor:
    non_zero_mask = torch.abs(targets) > zero_tol
    return torch.abs((preds[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask])


def model_device(model: torch.nn.Module) -> torch.device:
    # assumes all weights are on the same device
    return next(model.parameters()).device

@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: Union[Iterable[BaseData], DataLoader], loss_fn: Callable, out_transform: Callable):
    model.eval()
    metrics_manager = MetricsManager()

    for batch in dataset:
        batch = batch.to(str(model_device(model)))
        pred = model(batch)
        loss = loss_fn(pred.squeeze(), batch.y.squeeze())
        metrics_manager.update(out_transform(pred.squeeze()), out_transform(batch.y.squeeze()), loss.item())

    return metrics_manager.compute()