from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag, masked_select_nnz
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')


class RGATConv(MessagePassing):
    _alpha: OptTensor
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 num_relations: int,
                 heads: int = 1,
                 window_past: int = 3,
                 window_future: int = 3,
                 encoding: str = None,
                 concat: bool = True,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 device: str = None,
                 add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(RGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.heads = heads
        self.window_past = window_past
        self.window_future = window_future
        self.encoding = encoding
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.device = device
        if encoding == "relational":
            self.encoding_layer_weight = Parameter(torch.Tensor(num_relations, 1))
            self.encoding_layer_bias = Parameter(torch.Tensor(num_relations, 1))
        elif encoding == "relative":
            self.encoding_layer_weight = Parameter(torch.Tensor(1, 1))
            self.encoding_layer_bias = Parameter(torch.Tensor(1, 1))
        elif encoding == "multi":
            self.encoding_layer_weight = Parameter(torch.Tensor(num_relations, self.heads))
            self.encoding_layer_bias = Parameter(torch.Tensor(num_relations, self.heads))

        if isinstance(in_channels, int):
            self.lin_l = Parameter(torch.Tensor(num_relations, heads, out_channels))
            self.lin_r = self.lin_l

        self.att_l = Parameter(torch.Tensor(num_relations, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(num_relations, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l)
        glorot(self.lin_r)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)
        if self.encoding == "relational" or self.encoding == "relative" or self.encoding == "multi":
            glorot(self.encoding_layer_weight)
            glorot(self.encoding_layer_bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_type: Optional[torch.Tensor] = None,
                size: Size = None, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x = x.view(-1, self.heads, self.out_channels)
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_type=edge_type, size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type: Tensor, index: Tensor, edge_index_i: Tensor,
                edge_index_j: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        x_i = x_i * self.lin_l[edge_type]  # edge, head, channel
        x_j = x_j * self.lin_r[edge_type]  # edge, head, channel
        alpha_i = (x_i * self.att_l[edge_type]).sum(-1)  # edge, head
        alpha_j = (x_j * self.att_r[edge_type]).sum(-1)
        alpha = alpha_i + alpha_j  # edge, head
        relative_index = torch.FloatTensor((edge_index_j - edge_index_i).cpu().numpy()).to(self.device).unsqueeze(-1)
        if self.encoding == "relational" or self.encoding == "multi":
            positional_encodings = self.encoding_layer_weight[edge_type] * relative_index + self.encoding_layer_bias[edge_type]  # edge, 1
            alpha += positional_encodings
        elif self.encoding == "relative":
            positional_encodings = self.encoding_layer_weight * relative_index + self.encoding_layer_bias  # edge, 1
            alpha += positional_encodings
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        return x_j * alpha.unsqueeze(-1)  # edge, head, channel

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
