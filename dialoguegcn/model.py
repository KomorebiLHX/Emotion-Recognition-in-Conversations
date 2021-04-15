import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GraphConv
from RGCNConv import RGCNConv
import numpy as np
from utils import batch_graphify, classify_node_features


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim is not None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)  # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim
        return attn_pool, alpha


class MaskedEdgeAttention(nn.Module):
    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper.
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """
        super(MaskedEdgeAttention, self).__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> seq, batch, 2*D_e
        """
        scale = self.scalar(M)  # seq, batch, seq
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, seq, seq
        mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()  # never require gradient
        mask_copy = Variable(torch.zeros(alpha.size())).detach()
        edge_ind_ = []
        for i, j in enumerate(edge_ind):
            for x in j:
                edge_ind_.append([i, x[0], x[1]])
        edge_ind_ = np.array(edge_ind_).transpose()  # 3, seq
        mask[edge_ind_] = 1
        mask_copy[edge_ind_] = 1
        masked_alpha = alpha * mask
        _sums = masked_alpha.sum(-1, keepdim=True)
        scores = masked_alpha.div(_sums) * mask_copy  # can not divided by zero
        return scores


class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, max_seq_len, hidden_size=64, dropout=0.5,
                 no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()

        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.matchatt = MatchingAttention(num_features + hidden_size, num_features + hidden_size, att_type='general2')
        self.linear = nn.Linear(num_features + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        out = self.conv1(x, edge_index, edge_type, edge_norm)
        out = self.conv2(out, edge_index)
        emotions = torch.cat([x, out], dim=-1)
        log_prob = classify_node_features(emotions,
                                          seq_lengths,
                                          umask,
                                          self.matchatt,
                                          self.linear,
                                          self.dropout,
                                          self.smax_fc,
                                          nodal_attn,
                                          avec,
                                          self.no_cuda)
        return log_prob


class DialogueGCNModel(nn.Module):
    def __init__(self, base_model, D_m, D_e, graph_hidden_size, n_speakers, max_seq_len, window_past, window_future,
                 n_classes=7, listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5, nodal_attention=True, avec=False, no_cuda=False):
        super(DialogueGCNModel, self).__init__()
        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future
        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention
        self.graph_net = GraphNetwork(2 * D_e,
                                      n_classes,
                                      n_relations,
                                      max_seq_len,
                                      graph_hidden_size,
                                      dropout,
                                      self.no_cuda)
        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)
        self.edge_type_mapping = edge_type_mapping

    def forward(self, U, qmask, umask, seq_lengths):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, _ = self.lstm(U)  # emotions(seq, batch, 2 * D_e)
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(emotions,
                                                                                        qmask,
                                                                                        seq_lengths,
                                                                                        self.window_past,
                                                                                        self.window_future,
                                                                                        self.edge_type_mapping,
                                                                                        self.att_model,
                                                                                        self.no_cuda)
        log_prob = self.graph_net(features,
                                  edge_index,
                                  edge_norm,
                                  edge_type,
                                  seq_lengths,
                                  umask,
                                  self.nodal_attention,
                                  self.avec)
        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths
