import torch
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from typing import List, Tuple


def add_layer_norm(model: nn.Module, size: int) -> nn.Sequential:
    return nn.Sequential(model, nn.LayerNorm(size))


def make_mlp(in_size: int, latent_size: int, out_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_size, latent_size),
        nn.ReLU(),
        nn.Linear(latent_size, latent_size),
        nn.ReLU(),
        nn.Linear(latent_size, latent_size),
        nn.ReLU(),
        nn.Linear(latent_size, out_size),
    )


class GraphNetBlock(MessagePassing):
    r"""
    define message passing scheme
    # Math: \mathbf{x}_i^{\prime} = \mathbf{x}_i + \mathrm{MLP}_{\mathbf{\Theta}} \left(\mathbf{x}_i, \Sigma_{j \in \mathcal{N}(i)} \, \mathrm{MLP}_{\mathbf{\Theta}} \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right)\right),
    fitting the general framework
    # Math: \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i, \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}} \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),
    """
    
    def __init__(self, latent_size: int):
        super().__init__(aggr='add')
        # Node function
        # Math: \gamma_{\mathbf{\Theta}}\left(\mathbf{x}_i, \text{aggregated_messages}_i\right) = \mathbf{x}_i + \mathrm{MLP}_{\mathbf{\Theta}} \left(\mathbf{x}_i, \text{aggregated_messages}_i\right)
        # where x are the node attributes
        #       aggregated_messages are # Math: \text{aggregated_message}_i =\bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}} \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right)
        self.node_mlp = add_layer_norm(make_mlp(in_size=latent_size*2, latent_size=latent_size, out_size=latent_size), latent_size)
        
        # Define edge messages through the edge function
        # Math: \text{message}_{ij} = \phi_{\mathbf{\Theta}} \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) = \mathrm{MLP}_{\mathbf{\Theta}} \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right)
        # where x_i are the center nodes
        #       x_j are the neighboring nodes
        #       edge_attr are the edge attributes
        self.edge_mlp = add_layer_norm(make_mlp(in_size=latent_size*3, latent_size=latent_size, out_size=latent_size), latent_size)
        
    def forward(self, graph: BaseData) -> BaseData:
        # Update edge attributes
        edge_attr_updated = self.edge_updater(edge_index=graph.edge_index, edge_attr=graph.edge_attr, x=graph.x)

        # Update node attributes with messages=edge_attr_updated
        x_updated = self.propagate(graph.edge_index, x=graph.x, precomputed_messages=edge_attr_updated)

        # Skip connections
        edge_attr_updated += graph.edge_attr 
        x_updated += graph.x 
        return Data(x=x_updated, edge_attr=edge_attr_updated, edge_index=graph.edge_index)

    def message(self, precomputed_messages: Tensor) -> Tensor:
        return precomputed_messages

    def update(self, aggregated_messages: Tensor, x: Tensor) -> Tensor:
        r"""
        Node update function
        # Math: \gamma_{\mathbf{\Theta}}\left(\mathbf{x}_i, \text{aggregated_messages}_i\right) = \mathbf{x}_i + \mathrm{MLP}_{\mathbf{\Theta}} \left(\mathbf{x}_i, \text{aggregated_messages}_i\right)
        where x are the node attributes
              aggregated_messages are # Math: \text{aggregated_messages}_i =\bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}} \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right)
        """
        return self.node_mlp(torch.cat((x, aggregated_messages), dim=-1))
    
    def edge_update(self, edge_attr: Tensor, x_j: Tensor, x_i: Tensor) -> Tensor:
        r"""
        Define edge messages through the edge function
        # Math: \text{message}_{ij} = \phi_{\mathbf{\Theta}} \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) = \mathrm{MLP}_{\mathbf{\Theta}} \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right)
        where x_i are the center nodes
              x_j are the neighboring nodes
              e_{ij} are the edge attributes
        """
        return self.edge_mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))
    
    def __repr__(self):
        return (f'GraphNetBlock(\n'
                f'  node_mlp={self.node_mlp},\n'
                f'  edge_mlp={self.edge_mlp},\n'
                f'  aggr={self.aggr}\n'
                f')')


class Processor(nn.Module):
    def __init__(self, latent_size: int, num_message_passing_steps: int, dropout_rate: float):
        super().__init__()
        self.processors = nn.ModuleList(GraphNetBlock(latent_size) for _ in range(num_message_passing_steps))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, graph: BaseData) -> BaseData:
        for processor in self.processors:
            graph.x = self.dropout(graph.x) # apply dropout after encoder and between all processors
            graph = processor(graph)
        return graph

class Encoder(nn.Module):
    def __init__(self, num_node_features: int, num_edge_features: int, latent_size: int):
        super().__init__()
        self.node_encoder = add_layer_norm(make_mlp(in_size=num_node_features, latent_size=latent_size, out_size=latent_size), size=latent_size)
        self.edge_encoder = add_layer_norm(make_mlp(in_size=num_edge_features, latent_size=latent_size, out_size=latent_size), size=latent_size)

    def forward(self, graph: BaseData) -> BaseData:
        return Data(x=self.node_encoder(graph.x), edge_attr=self.edge_encoder(graph.edge_attr), edge_index=graph.edge_index)

class Decoder(nn.Module):
    def __init__(self, latent_size: int, out_size: int):
        super().__init__()
        self.node_decoder = make_mlp(in_size=latent_size, latent_size=latent_size, out_size=out_size)

    def forward(self, graph: BaseData) -> Tensor:
        return self.node_decoder(graph.x)

class GraphNet(nn.Module):
    def __init__(self, num_node_features: int, num_edge_features: int, latent_size: int, num_node_outputs: int, num_message_passing_steps: int, dropout_rate: float = 0.0):
        super().__init__()
        self.encoder = Encoder(num_node_features, num_edge_features, latent_size)
        self.processor = Processor(latent_size, num_message_passing_steps, dropout_rate)
        self.decoder = Decoder(latent_size, num_node_outputs)

    def forward(self, graph: BaseData) -> Tensor:
        graph = self.encoder(graph)
        graph = self.processor(graph)
        y = self.decoder(graph)
        return y
