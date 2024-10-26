from pathlib import Path
from typing import Dict, Any, Tuple
import datetime
import argparse
import json

import torch

from dataset import StructGraphDataset
from model import GraphNet
from metric import evaluate
from huggingface_io import huggingface_dataset_generator

def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments and return configuration dictionary.
    """
    parser = argparse.ArgumentParser(description='Evaluate GraphNet model on datasets')
    
    # Required arguments
    parser.add_argument('--dataset-path', type=Path, required=True,
                      help='Path to the cached dataset file')
    parser.add_argument('--checkpoint-path', type=Path, required=True,
                      help='Path to the model checkpoint file')
    
    # Optional arguments
    parser.add_argument('--device', type=str, 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run the model on')
    parser.add_argument('--extra-datasets', type=str,
                      help='JSON file containing extra datasets configuration')
    
    args = parser.parse_args()
    
    config = {
        "device": torch.device(args.device),
        "dataset_load_path": args.dataset_path,
        "checkpoint_load_path": args.checkpoint_path,
        "extra_datasets": []
    }
    
    if args.extra_datasets:
        with open(args.extra_datasets) as f:
            config["extra_datasets"] = json.load(f)
    
    return config

def load_datasets_from_disk(path: Path) -> Tuple[StructGraphDataset, StructGraphDataset, StructGraphDataset, Any, Any]:
    """
    Load datasets from a disk file.
    """
    print(f"Loading datasets from {path}")
    dataset = torch.load(path, weights_only=False)
    return dataset['train_dataset'], dataset['val_dataset'], dataset['test_dataset'], dataset['standardizer'], dataset['pre_transform']

def load_model(path: Path, device: torch.device) -> Tuple[GraphNet, Dict[str, Any]]:
    """
    Load the GraphNet model from a checkpoint file.
    """
    print(f"Loading model from {path}")
    checkpoint = torch.load(path, weights_only=False)
    training_config = checkpoint['config']
    print(f"Loaded model ({datetime.datetime.fromtimestamp(checkpoint['timestamp'])}, epoch {checkpoint['epoch']}) with training config: {training_config}")

    model = GraphNet(num_node_features=checkpoint['num_node_features'],
                     num_edge_features=checkpoint['num_edge_features'],
                     num_node_outputs=checkpoint['num_node_outputs'],
                     latent_size=training_config['latent_size'],
                     num_message_passing_steps=training_config['num_message_passing_steps'],
                     dropout_rate=training_config['dropout_rate']
                     ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, training_config

def format_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format dictionary values for pretty printing.
    """
    return {k: f"{v:.2e}" if isinstance(v, float) else v for k, v in d.items()}

def main():
    config = parse_args()
    
    train_dataset, val_dataset, test_dataset, standardizer, pre_transform = load_datasets_from_disk(config['dataset_load_path'])
    train_dataset, val_dataset, test_dataset = (tuple(data.to(config['device']) for data in dataset) for dataset in (train_dataset, val_dataset, test_dataset))
    destandardizer = standardizer.make_destandardizer('y')
    
    model, training_config = load_model(config['checkpoint_load_path'], config['device'])

    print(f"Train statistics:      {format_dict(evaluate(model, train_dataset, training_config['loss'], destandardizer))}")
    print(f"Validation statistics: {format_dict(evaluate(model, val_dataset, training_config['loss'], destandardizer))}")
    print(f"Test statistics:       {format_dict(evaluate(model, test_dataset, training_config['loss'], destandardizer))}")

    for extra_dataset in config["extra_datasets"]:
        dataset_generator = lambda: huggingface_dataset_generator(extra_dataset["path"], extra_dataset["group"])
        dataset = StructGraphDataset(Path(".cache")/extra_dataset["path"]/extra_dataset["group"], 
                                   dataset_generator, pre_transform=pre_transform)
        dataset = [standardizer(data) for data in dataset]
        print(f"{extra_dataset['name']} statistics: {format_dict(evaluate(model, dataset, training_config['loss'], destandardizer))}")

if __name__ == '__main__':
    main()