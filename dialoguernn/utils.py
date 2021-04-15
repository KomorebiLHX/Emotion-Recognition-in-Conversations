import torch
from dataset import IEMOCAPDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
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
    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False, cuda=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer is not None
    if train:
        print("\t* Training epoch {}:".format(epoch))
        model.train()
    else:
        print("\t* Validating epoch {}:".format(epoch))
        model.eval()
    for data in tqdm(dataloader):
        if train:
            optimizer.zero_grad()
        # textf -> utterance
        # visuf -> visual
        # acouf -> audio
        # qmask -> speaker mask
        # umask -> utterance mask
        # label -> target
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        log_prob, alpha, alpha_f, alpha_b = model(textf, qmask, umask, att2=True)  # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    print("\t-> avg_loss: {:.4f}, avg_accuracy: {:.4f}, avg_fscore: {:.4f}".format(avg_loss, avg_accuracy, avg_fscore))
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]
