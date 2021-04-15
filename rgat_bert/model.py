import torch
import torch.nn as nn
from transformers import BertModel
from FastRGATConv import RGATConv
from utils import get_attention_mask, batch_graphify


class RGATmodel(nn.Module):
    def __init__(self, batch_size, dimension, n_classes, n_relations, n_layers, window_past, window_future, encoding, device):
        super(RGATmodel, self).__init__()
        self.batch_size = batch_size
        self.dimension = dimension
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.window_past = window_past
        self.window_future = window_future
        self.encoding = encoding
        self.device = device
        self.edge_type_mapping = {'000': 0,
                                  '110': 0,
                                  '001': 1,
                                  '111': 1,
                                  '010': 2,
                                  '100': 2,
                                  '011': 3,
                                  '101': 3}
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.rgat = nn.ModuleList([RGATConv(dimension, dimension // 8, n_relations, heads=8, window_past=window_past, window_future=window_future, encoding=encoding, device=device) for i in range(n_layers)])
        self.layernorm = nn.ModuleList([nn.LayerNorm(dimension) for i in range(n_layers)])
        self.classifier = nn.Sequential(nn.Linear(dimension * 2, dimension // 2),
                                        nn.ReLU(),
                                        nn.Linear(dimension // 2, n_classes))
        self.encoding_layer = nn.Embedding(110, dimension)

    def forward(self, ids, qmask, lengths):
        """
        ids -> seq, batch, dim_utterance
        qmask -> seq_len, batch, party
        """
        ids = ids.reshape(-1, ids.size(2))
        attention_mask = get_attention_mask(ids, self.device)
        bert_output = self.bert(ids, attention_mask).pooler_output  # seq*batch, dim
        features, edge_index, edge_type = batch_graphify(bert_output,
                                                         qmask,
                                                         lengths,
                                                         self.window_past,
                                                         self.window_future,
                                                         self.dimension,
                                                         self.edge_type_mapping,
                                                         self.device)
        if self.encoding == "absolute":
            features += self.encoding_layer(torch.LongTensor([i for i in range(len(features))]).to(self.device))
        graph_output_0 = self.rgat[0](features, edge_index, edge_type)
        output_norm_0 = self.layernorm[0](graph_output_0)
        graph_output_1 = self.rgat[1](output_norm_0, edge_index, edge_type)
        output_norm_1 = self.layernorm[1](graph_output_1)
        graph_output_2 = self.rgat[2](output_norm_1, edge_index, edge_type)
        output_norm_2 = self.layernorm[2](graph_output_2)
        output_all = torch.cat([features, output_norm_2], dim=-1)
        logits = self.classifier(output_all)
        return logits
