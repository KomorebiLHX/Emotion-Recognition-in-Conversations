import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_IEMOCAP_loaders, train_or_eval_graph_model
from model import RGATmodel

device = "cuda" if torch.cuda.is_available() else "cpu"
n_classes = 6
n_epochs = 40
n_relations = 4
n_layers = 3
window_past = 5
window_future = 5
batch_size = 1
lr_bert = 4e-5
lr = 2e-3
eta_min = 2e-5
T_max = 8
dim = 768
max_len_conversation = 110
max_len_utterance = 126

encoding = "multi"

if encoding is not None:
    print('Positional Encoding: ' + encoding)
else:
    print('Positional Encoding is None')

if device == "cuda":
    print('Running on GPU')
else:
    print('Running on CPU')

print(20 * '=', "Preprocessing data", 20 * '=')
print("\t* building iterators")
train_loader, valid_loader, test_loader = get_IEMOCAP_loaders("IEMOCAP_features_bert.pkl",
                                                              valid=0.0,
                                                              batch_size=batch_size,
                                                              num_workers=0)
print(20 * '=', "Training data", 20 * '=')
model = RGATmodel(batch_size=batch_size,
                  dimension=dim,
                  n_classes=n_classes,
                  n_relations=n_relations,
                  n_layers=n_layers,
                  window_past=window_past,
                  window_future=window_future,
                  encoding=encoding,
                  device=device)
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.bert.parameters(), 'lr': lr_bert},
    {'params': model.rgat.parameters()},
    {'params': model.layernorm.parameters()},
    {'params': model.classifier.parameters()},
    {'params': model.encoding_layer.parameters()}], lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
all_fscore = []
for e in range(n_epochs):
    train_loss, train_acc, train_fscore = \
        train_or_eval_graph_model(model, loss_function, train_loader, e, device, optimizer=optimizer, train=True)
    test_loss, test_acc, test_fscore = \
        train_or_eval_graph_model(model, loss_function, test_loader, e, device, optimizer=None, train=False)
    scheduler.step()
    all_fscore.append(test_fscore)
    if train_fscore >= 100.0:
        break
print('\t-> Finished, Best F1 Score:', max(all_fscore))
