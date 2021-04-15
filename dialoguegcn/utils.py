import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import IEMOCAPDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def seed_everything(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use determinisitic algorithm


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def edge_perms(l, window_past, window_future):
    """
    Method to construct the edges considering the past and future window.
    l -> sequence length
    """

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()
        # select utterances in the window
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]
        # add edges(verticle1, verticle2) in perms
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)

    return list(all_perms)


def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    """
    Method to prepare the data format required for the GCN network. Pytorch geometric puts all nodes for classification
    in one single graph. Following this, we create a single graph for a mini-batch of dialogue instances. This method
    ensures that the various graph indexing is properly carried out so as to make sure that, utterances (nodes) from
    each dialogue instance will have edges with utterances in that same dialogue instance, but not with utternaces
    from any other dialogue instances in that mini-batch.
    features -> seq, batch, 2*D_m
    qmask -> seq, batch, party
    """

    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []
    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))

    # scores are the edge weights
    scores = att_model(features, lengths, edge_ind)  # scores(batch, seq, seq)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

        perms1 = edge_perms(lengths[j], window_past, window_future)  # local
        perms2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms1]  # global
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))

        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])

            speaker0 = (qmask[item1[0], j, :] == 1).nonzero(as_tuple=False)[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero(as_tuple=False)[0][0].tolist()

            if item1[0] < item1[1]:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])

    node_features = torch.cat(node_features, dim=0)
    # tensor(length*batch, 2*D_e), length is different between different conversations
    edge_index = torch.stack(edge_index).transpose(0, 1)  # tensor(edge, 2)
    edge_norm = torch.stack(edge_norm)  # tensor(edge)
    edge_type = torch.tensor(edge_type)  # tensor(edge)

    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            #if torch.cuda.is_available():
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    """
    Method to obtain attentive node features over the graph convoluted features, as in Equation 4, 5, 6. in the paper.
    """

    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()

    max_len = max(seq_lengths)

    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda)
                            for s, l in zip(start.data.tolist(),
                                            input_conversation_length.data.tolist())], 0).transpose(0, 1)

    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:, 0, :])

    att_emotions = torch.cat(att_emotions, dim=0)

    return att_emotions


def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer, nodal_attn, avec, no_cuda):
    """
    Function for the final classification, as in Equation 7, 8, 9. in the paper.
    """
    emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
    hidden = F.relu(linear_layer(emotions))
    hidden = dropout_layer(hidden)
    hidden = smax_fc_layer(hidden)
    log_prob = F.log_softmax(hidden, 2)
    log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
    return log_prob


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    assert not train or optimizer is not None
    if train:
        print("\t* Training epoch {}:".format(epoch))
        model.train()
    else:
        print("\t* Validating epoch {}:".format(epoch))
        model.eval()

    seed_everything()
    for data in tqdm(dataloader):
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        lengths = [torch.nonzero((umask[j] == 1), as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)

        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    print("\t-> avg_loss: {:.4f}, avg_accuracy: {:.4f}, avg_fscore: {:.4f}".format(avg_loss, avg_accuracy, avg_fscore))
    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el
