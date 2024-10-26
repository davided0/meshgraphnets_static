import time
from typing import Dict, Iterable, List, Callable, Any, Tuple, Sequence
from pathlib import Path
from io import StringIO
import sys
import json

import torch
from torch_geometric.transforms import Compose, ToDevice, Cartesian, Distance, BaseTransform
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam

from model import GraphNet
from metric import MetricsManager, evaluate
from dataset import stratified_split, standardize, AugmentEdges, SimulationCoords, StructGraphDataset, Standardizer
from huggingface_io import huggingface_dataset_generator

class StdoutRedirect:
    def __init__(self, line_callback: Callable[[str], None], reset_buffer: bool = True):
        self.line_callback = line_callback
        self.reset_buffer = reset_buffer
        self.original_stdout = sys.stdout
        self.output_buffer = StringIO()

    def __enter__(self):
        """Set up the stdout redirection when entering the context."""
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore the original stdout when exiting the context."""
        sys.stdout = self.original_stdout
        self.output_buffer.close()

    def write(self, text: str):
        self.original_stdout.write(text)
        self.output_buffer.write(text)
        if text.endswith('\n'):
            self.line_callback(self.output_buffer.getvalue().strip())
            if self.reset_buffer:
                self.output_buffer = StringIO()

def log_metrics(logger: SummaryWriter, train_metrics: Dict[str, float], val_metrics: Dict[str, float], extra_metrics: Tuple[Dict[str, float], ...], epoch: int, lr: float) -> None:
    """Log training and validation metrics."""
    logger.add_scalars('Losses', {"train": train_metrics["loss"], "val": val_metrics["loss"]}, global_step=epoch)
    logger.add_scalar('LR', lr, global_step=epoch)

    relative_errors = {
        **{f'train_{k}': v for k, v in train_metrics.items() if k.endswith("relative_error")},
        **{f'val_{k}': v for k, v in val_metrics.items() if k.endswith("relative_error")}
    }

    for i, metrics in enumerate(extra_metrics):
        relative_errors.update({
            f'extra_{i}_{k}': v for k, v in metrics.items() if k.endswith("relative_error")
        })

    logger.add_scalars('Relative_Error', relative_errors, global_step=epoch)
    
def print_epoch_summary(epoch: int, num_epochs: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], lr: float, epoch_start: float, training_start: float) -> None:
    """Print a summary of the epoch results."""
    print(f'epoch {epoch:04d}/{num_epochs}, '
          f'loss: {train_metrics["loss"]:.2e}/{val_metrics["loss"]:.2e}, '
          f'mean_pointwise_relative_error: {train_metrics["mean_pointwise_relative_error"]:.2e}/{val_metrics["mean_pointwise_relative_error"]:.2e}, '
          f'max_pointwise_relative_error: {train_metrics["max_pointwise_relative_error"]:.2e}/{val_metrics["max_pointwise_relative_error"]:.2e}, '
          f'mean_batch_peak_relative_error: {train_metrics["mean_batch_peak_relative_error"]:.2e}/{val_metrics["mean_batch_peak_relative_error"]:.2e}, '
          f'max_batch_peak_relative_error: {train_metrics["max_batch_peak_relative_error"]:.2e}/{val_metrics["max_batch_peak_relative_error"]:.2e}, '
          f'lr:{lr:.2e}, '
          f'epoch time:{time.time()-epoch_start:.2f}s, '
          f'total time:{time.strftime("%Hh%Mm%Ss", time.gmtime(time.time()-training_start))}')

def save_checkpoint(model: torch.nn.Module, path: Path, epoch: int, config: Dict[str, Any], dataset_info: Dict[str, int]) -> None:
    """Save a checkpoint of the model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'config': config,
        'num_node_features': dataset_info['num_node_features'],
        'num_edge_features': dataset_info['num_edge_features'],
        'num_node_outputs': dataset_info['num_node_outputs'],
        'timestamp': time.time(),
    }, path)

def save_dataset(train_dataset: Sequence[BaseData], val_dataset: Sequence[BaseData], test_dataset: Sequence[BaseData], standardizer: Standardizer, pre_transform: BaseTransform, save_path: Path) -> None:
    """Save the prepared datasets."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'standardizer': standardizer,
        'pre_transform': pre_transform,
    }, save_path)

def get_dataset_info(sample: BaseData) -> Dict[str, Any]:
    """Extract dataset information from a sample."""
    num_features = lambda x: x.shape[-1] if x.dim() > 1 else 1
    return {
        'num_node_features': num_features(sample.x),
        'num_edge_features': num_features(sample.edge_attr),
        'num_node_outputs': num_features(sample.y)
    }

def make_model(dataset_info: Dict[str, Any], config: Dict[str, Any]) -> torch.nn.Module:
    """Create the Graph Neural Network model."""
    return GraphNet(
        num_node_features=dataset_info['num_node_features'],
        num_edge_features=dataset_info['num_edge_features'],
        num_node_outputs=dataset_info['num_node_outputs'],
        latent_size=config['latent_size'],
        num_message_passing_steps=config['num_message_passing_steps'],
        dropout_rate=config['dropout_rate']
    ).to(config['device'])

def make_scheduler(optimizer: Optimizer, config: Dict[str, Any]) -> lr_scheduler.LRScheduler:
    """Create the learning rate scheduler."""
    return lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(
            config['min_lr'] / config['initial_lr'],
            config['decay_rate'] ** (step / config['decay_steps'])
        )
    )

def get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the device of the model assuming all weights reside on the same device."""
    return next(model.parameters()).device

def load_dataset_groups(dataset_path: str, groups: Iterable[str], pre_transform: Callable) -> Iterable[Dataset]:
    """Create datasets for each group."""
    return [
        StructGraphDataset(
            str(Path(".cache") / dataset_path / group),
            lambda: huggingface_dataset_generator(dataset_path, group),
            pre_transform=pre_transform
        )
        for group in groups
    ]

def make_transform(config: Dict[str, Any]) -> BaseTransform:
    """Create a composition of transforms based on the given parameters."""
    transforms = [
        ToDevice(config['device']),
        AugmentEdges(config['edge_augmentation']),
        Cartesian(norm=False),
        Distance(norm=False)
    ]
    if config['simulation_coords']:
        transforms.append(SimulationCoords())
    return Compose(transforms)

def train_epoch(model: torch.nn.Module, dataset: DataLoader, loss_fn: Callable, out_transform: Callable, optimizer: Optimizer) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    metrics_manager = MetricsManager()

    for batch in dataset:
        batch = batch.to(get_model_device(model))
        optimizer.zero_grad()

        pred = model(batch)
        loss = loss_fn(pred.squeeze(), batch.y)

        with torch.no_grad():
            metrics_manager.update(out_transform(pred), out_transform(batch.y), loss)

        loss.backward()
        optimizer.step()

    return metrics_manager.compute()
    
def train(config: Dict[str, Any], logger: SummaryWriter) -> None:
    """Run the main training loop."""
    print("Config:")
    for key, value in config.items():
        print(f" - {key}: {value}")

    pre_transform = make_transform(config)
    dataset_groups = load_dataset_groups(config["dataset_path"], config['training_groups'], pre_transform)
    train_dataset, val_dataset, test_dataset = stratified_split(dataset_groups, split_ratios=config['train_test_val_split'])
    train_dataset, val_dataset, test_dataset, standardizer = standardize(train_dataset, val_dataset, test_dataset)
    train_dataset, val_dataset, test_dataset = (tuple(data.cpu() for data in dataset) for dataset in (train_dataset, val_dataset, test_dataset))
    destandardizer = standardizer.make_destandardizer('y')

    save_dataset(train_dataset, val_dataset, test_dataset, standardizer, pre_transform, config['dataset_save_path'])
    print(f"Dataset size: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")
    print(f'Standardizer {standardizer}')

    extra_groups = load_dataset_groups(config["dataset_path"], config['extra_groups'], pre_transform)
    extra_groups = tuple([standardizer(data) for data in dataset] for dataset in extra_groups)

    dataset_info = get_dataset_info(train_dataset[0])
    model = make_model(dataset_info, config)

    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = Adam(model.parameters(), lr=config['initial_lr'])
    scheduler = make_scheduler(optimizer, config)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    training_start = time.time()
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        train_metrics = train_epoch(model, train_loader, config['loss'], destandardizer, optimizer)
        val_metrics = evaluate(model, val_dataset, config['loss'], destandardizer)
        extra_metrics = tuple(evaluate(model, extra_group, config['loss'], destandardizer) for extra_group in extra_groups)

        lr = scheduler.get_last_lr()[-1]
        print_epoch_summary(epoch, config['num_epochs'], train_metrics, val_metrics, lr, epoch_start, training_start)
        log_metrics(logger, train_metrics, val_metrics, extra_metrics, epoch, lr)
        if epoch % config['checkpoint_freq'] == 0:
            save_checkpoint(model, config['checkpoint_save_path'], epoch, config, dataset_info)
        
        scheduler.step()

    save_checkpoint(model, config['checkpoint_save_path'], config['num_epochs'], config, dataset_info)

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load the configuration from a file."""

    with open(config_path, 'r') as file:
        config = json.load(file)
    
    loss_functions = {
        'mse': torch.nn.functional.mse_loss,
        'l1': torch.nn.functional.l1_loss,
    }
    config['loss'] = loss_functions[config['loss']]
    return config

def main(config_path: Path) -> None:
    config = load_config(config_path)
    logger = SummaryWriter(config['log_folder'])
    text_logger=lambda text: logger.add_text("cout", text, global_step=0)
    with StdoutRedirect(line_callback=text_logger, reset_buffer=False):
        train(config, logger)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test.py <path_to_config.json>")
        sys.exit(1)
    main(config_path=Path(sys.argv[1]))
