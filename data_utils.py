# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from config import letters, categorys, DEVICE, DATA_PATH, BATCH_SIZE

def read_names(file_path=DATA_PATH):
    names, labels = [], []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            if len(line) <= 5:
                continue
            name, label = line.strip().split('\t')
            names.append(name)
            labels.append(label)
    return names, labels

class NameDataset(Dataset):
    def __init__(self, names, labels):
        self.names = names
        self.labels = labels

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.labels[idx]

        # one-hot 编码
        tensor = torch.zeros(len(name), len(letters))
        for i, char in enumerate(name):
            tensor[i][letters.find(char)] = 1

        return tensor, torch.tensor(categorys.index(label), dtype=torch.long)

def get_dataloader():
    names, labels = read_names()
    dataset = NameDataset(names, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)






