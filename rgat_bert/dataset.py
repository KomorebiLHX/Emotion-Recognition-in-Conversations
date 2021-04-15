import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd


class IEMOCAPDataset(Dataset):

    def __init__(self, path, train):
        self.videoIDs, self.videoTextIDs, self.videoSpeakers, self.videoLabels, self.trainVid, self.testVid = \
            pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.stack(self.videoTextIDs[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        d = pd.DataFrame(data)
        return [pad_sequence(d[i]) if i < 2 else pad_sequence(d[i], True) if i < 4 else d[i].tolist() for i in d]


if __name__ == "__main__":
    dataset = IEMOCAPDataset("IEMOCAP_features_bert.pkl", train=True)







