"""EdgeConv–GLADC and its neighbor, standardized and fraction controls."""

import torch
from torch import nn
from torch.func import functional_call
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import MessagePassing

from .edgeconv import EdgeConvAE, _EncoderBlock
from .netge import dense_nodes, log_fractions


GLADC_VARIANTS = ('original', 'neighbor', 'standardized', 'fraction')


class _NeighborEncoderBlock(_EncoderBlock):
    """Add h_j-h_i to messages, initializing the new columns at zero."""

    def __init__(self, block):
        MessagePassing.__init__(self, aggr='mean')
        self.residual, self.mlp = block.residual, block.mlp
        linear = self.mlp[0]
        width = linear.in_features - 3
        weight = linear.weight.detach()
        linear.weight = nn.Parameter(torch.cat([
            weight[:, :width], weight.new_zeros((linear.out_features, width)), weight[:, width:],
        ], dim=1))
        linear.in_features += width

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))


class GLADC(EdgeConvAE):
    """Add a shared 2->128->128 graph projection for cycle and contrast losses."""

    def __init__(self, variant='original'):
        super().__init__()
        self.contrastive_projection = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128))
        self.variant = variant
        if variant == 'neighbor':
            self.encoder_blocks = nn.ModuleList([_NeighborEncoderBlock(block) for block in self.encoder_blocks])
        if variant in ('standardized', 'fraction'):
            for name, size in (('node', 1), ('edge', 3)):
                self.register_buffer(name + '_center', torch.zeros(size))
                self.register_buffer(name + '_scale', torch.ones(size))

    def prepare(self, jets):
        """Transform real inputs once; cycle predictions are already in these coordinates."""
        if self.variant in ('original', 'neighbor'):
            return jets
        prepared = jets.clone()
        nodes = jets.x
        if self.variant == 'fraction':
            x, mask = dense_nodes(jets)
            nodes = log_fractions(x, mask)[mask]
        prepared.x = (nodes - self.node_center) / self.node_scale
        prepared.edge_attr = (jets.edge_attr - self.edge_center) / self.edge_scale
        return prepared

    @torch.no_grad()
    def perturbed_graph(self, batch):
        """Add Gaussian noise to encoder weights; leave projection weights unchanged."""
        hidden = batch.x
        for block in self.encoder_blocks:
            parameters = {}
            for name, parameter in block.named_parameters():
                value = parameter.detach()
                std = value.std() if value.numel() > 1 else value.new_zeros(())
                parameters[name] = value + torch.randn_like(value) * std
            hidden = functional_call(block, parameters, (hidden, batch.edge_index, batch.edge_attr))
        return self.contrastive_projection(global_max_pool(hidden, batch.batch))
