import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
from torch.utils.data import ConcatDataset, Subset, random_split, Dataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import BaseData
import random
import hashlib
from pathlib import Path

class Standardizer(BaseTransform, ABC):
    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        pass

    @abstractmethod
    def __call__(self, data: BaseData) -> BaseData:
        pass

    @abstractmethod
    def make_destandardizer(self, attr: str) -> BaseTransform:
        pass

class Destandardizer(BaseTransform):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.std = std

    @torch.no_grad()
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.to(data.device)
        self.std = self.std.to(data.device)
        return data * self.std + self.mean

class ZScoreNormalize(Standardizer):
    def __init__(self, attrs: List[str]):
        self.attrs = attrs
        self.mean = {}
        self.std = {}

    @torch.no_grad()
    def fit(self, dataset: Sequence[BaseData]) -> None:
        for attr in self.attrs:
            # TODO: avoid storing all data points in memory
            all_attr = torch.cat([getattr(data, attr) for data in dataset])
            self.mean[attr] = all_attr.mean(dim=0, keepdim=True)
            self.std[attr] = all_attr.std(dim=0, keepdim=True)
    
    @torch.no_grad()
    def std_with_eps(self, attr: str, eps: float = 1e-8) -> torch.Tensor:
        return torch.maximum(self.std[attr], torch.tensor(eps))
    
    @torch.no_grad()
    def __call__(self, data: BaseData) -> BaseData:
        for attr in self.attrs:
            orig = getattr(data, attr)
            normalized = (orig - self.mean[attr]) / self.std_with_eps(attr)
            setattr(data, attr, normalized)
        return data
    
    @torch.no_grad()
    def make_destandardizer(self, attr: str) -> Destandardizer:
        return Destandardizer(self.mean[attr], self.std[attr])

    def __repr__(self):
        return f"ZScoreNormalize(attrs={self.attrs}, mean={self.mean}, std={self.std})"

def cat_or_new(existing: Optional[torch.Tensor], new: torch.Tensor, **kwargs) -> torch.Tensor:
    if existing is None:
        return new
    new = new.to(existing.device)
    return torch.cat((existing, new), **kwargs)

class AugmentEdges(BaseTransform):
    def __init__(self, augmentation_factor: float):
        self.augmentation_factor = augmentation_factor

    def __call__(self, data: Data) -> Data:
        if self.augmentation_factor <= 0 or data.num_nodes is None:
            return data

        num_nodes = data.num_nodes
        num_edges = data.num_edges
        num_new_edges = self._calculate_num_new_edges(num_edges)
        
        existing_edges = self._get_existing_edges(data)
        new_edges = self._generate_new_edges(num_nodes, num_new_edges, existing_edges)
        data = self._append_new_edges(data, new_edges, num_new_edges)

        self._validate_data(data, num_edges, num_new_edges)
        return data

    def _calculate_num_new_edges(self, num_edges: int) -> int:
        num_new_edges = int(num_edges * self.augmentation_factor)
        return num_new_edges + (num_new_edges % 2)

    def _get_existing_edges(self, data: Data) -> set:
        return set(map(tuple, data.edge_index.t().tolist())) if data.edge_index is not None else set()

    def _generate_new_edges(self, num_nodes: int, num_new_edges: int, existing_edges: set) -> List[Tuple[int, int]]:
        new_edges = []
        while len(new_edges) < num_new_edges:
            source, target = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            if source != target and (source, target) not in existing_edges and (target, source) not in existing_edges:
                new_edges.extend(((source, target), (target, source)))
                existing_edges.update(((source, target), (target, source)))
        return new_edges

    def _append_new_edges(self, data: Data, new_edges: List[Tuple[int, int]], num_new_edges: int) -> Data:
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        data.edge_index = cat_or_new(getattr(data, 'edge_index', None), new_edges_tensor, dim=1)
        self._add_augmentation_indicator(data, num_new_edges)
        return data

    def _add_augmentation_indicator(self, data: Data, num_new_edges: int) -> None:
        assert data.edge_index is not None, "AugmentEdges should mark existing edges"
        is_augmented = torch.zeros(data.num_edges, 1, dtype=torch.float, device=data.edge_index.device)
        is_augmented[-num_new_edges:] = 1.0
        
        assert getattr(data, 'edge_attr') is None, "AugmentEdges should be applied before any other edge attribute"
        data.edge_attr = is_augmented

    def _validate_data(self, data: Data, original_num_edges: int, num_new_edges: int) -> None:
        assert data.edge_index is not None, "Edges not set after AugmentEdges"
        assert data.edge_attr is not None, "Edge attributes not set after AugmentEdges"
        assert data.edge_index.shape[1] == original_num_edges + num_new_edges, "Incorrect number of edges after augmentation"
        assert data.edge_attr.shape[0] == data.edge_index.shape[1], "Edge attributes do not match the number of edges"

    def __repr__(self):
        return f'{self.__class__.__name__}(augmentation_factor={self.augmentation_factor})'

class SimulationCoords(BaseTransform):
    def __call__(self, data: Data) -> Data:
        if data.pos is None:
            return data

        X = data.pos
        X -= X.mean(dim=-2, keepdim=True)  # make shift invariant

        # Make rotation invariant
        _, V = torch.linalg.eigh(X.t() @ X)
        X @= V

        data.x = cat_or_new(getattr(data, 'x', None), X, dim=1)
        return data

def to_simulation_graph(item: Dict[str, Any]) -> Data:
    def cells_to_edges(cells: Sequence[Sequence[int]]) -> torch.Tensor:
        edge_pairs = [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)]
        cells_tensor = torch.tensor(cells, dtype=torch.long)
        edges = torch.cat([cells_tensor[:, [i, j]] for i, j in edge_pairs])
        return edges.t()

    node_pos = torch.tensor(item['mesh_coords'], dtype=torch.float32)
    node_types = torch.nn.functional.one_hot(torch.tensor(item['node_types'])).float()
    edge_index = cells_to_edges(item['mesh_cells'])
    von_mises = torch.tensor(item['von_mises'], dtype=torch.float)
    dataset_name = item['dataset_name']
    
    node_features = node_types
    output_features = von_mises
    return Data(pos=node_pos, x=node_features, edge_index=edge_index, y=output_features, dataset_name=dataset_name)

class StructGraphDataset(InMemoryDataset):
    def __init__(self, root: str, dataset_generator: Any, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.dataset_generator = dataset_generator
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.load(self.processed_paths[0])
        print(f'Loaded dataset with pre-transformation {Path(self.processed_dir) / "pre_transform.pt"}')

    def _hash_pre_transform(self) -> str:
        return hashlib.sha256(str(self.pre_transform).encode()).hexdigest()
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, f'processed_{self._hash_pre_transform()}')
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['dataset.pt']
    
    def process(self) -> None:
        data_list = (to_simulation_graph(item) for item in self.dataset_generator())
        if self.pre_transform is not None:
            data_list = (self.pre_transform(item) for item in data_list)
        self.save(list(data_list), self.processed_paths[0])

def stratified_split(dataset_groups: Iterable[Dataset], split_ratios: Sequence[Union[int, float]]):
    # split each group
    splits = (random_split(dataset_group, split_ratios) for dataset_group in dataset_groups)
    # concat each split across all groups
    train_dataset, val_dataset, test_dataset = (ConcatDataset(split) for split in zip(*splits))
    # shuffle
    train_dataset, val_dataset, test_dataset = (Subset(dataset, torch.randperm(len(dataset)).tolist()) for dataset in (train_dataset, val_dataset, test_dataset))

    # load transformed dataset for efficiency
    train_dataset, val_dataset, test_dataset = ([data for data in dataset] for dataset in (train_dataset, val_dataset, test_dataset))
    return train_dataset, val_dataset, test_dataset

def standardize(train_dataset: Sequence[BaseData], val_dataset: Sequence[BaseData], test_dataset: Sequence[BaseData]):
    # standardize
    standardizer = ZScoreNormalize(['x', 'y', 'edge_attr'])
    standardizer.fit(train_dataset)
    train_dataset, val_dataset, test_dataset = ([standardizer(data) for data in dataset] for dataset in (train_dataset, val_dataset, test_dataset))
    return train_dataset, val_dataset, test_dataset, standardizer
