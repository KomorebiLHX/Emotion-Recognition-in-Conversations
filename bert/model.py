import torch
import torch.nn as nn
from transformers import BertModel
from utils import get_attention_mask, batch_graphify


class Bert_base(nn.Module):
    def __init__(self, batch_size, dimension, n_classes, device):
        super(Bert_base, self).__init__()
        self.batch_size = batch_size
        self.dimension = dimension
        self.n_classes = n_classes
        self.device = device
        self.bert = BertModel.from_pretrained("./bert-base-uncased")
        self.classifier = nn.Sequential(nn.Linear(dimension, dimension // 2, bias=True),
                                        nn.ReLU(),
                                        nn.Linear(dimension // 2, n_classes, bias=True))

    def forward(self, ids):
        """
        ids -> seq, batch, dim_utterance
        qmask -> seq_len, batch, party
        """
        ids = ids.reshape(-1, ids.size()[2])
        attention_mask = get_attention_mask(ids, self.device)
        bert_output = self.bert(ids, attention_mask).pooler_output  # seq*batch, dim
        logits = self.classifier(bert_output)
        return logits
