import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset


class except_Dataset(Dataset):
    training_file = "train.pkl"
    test_file = "test.pkl"

    def __init__(self, train=True):
        file_root = '/except_2/'
        if train:
            data_file = file_root + self.training_file
        else:
            data_file = file_root + self.test_file
        with open(data_file, "rb") as fp:
            entry = pickle.load(fp)
            self.indices = entry["indices"]
            self.data = entry["data"]
            self.labels = entry["labels"]

        with open(file_root + "standard_deviation.pkl", "rb") as fp:
            entry = pickle.load(fp)
            self.mean = entry["mean"]
            self.std = entry["std"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].copy()  # (seq_length, feat_dim) array
        x[:, 6:28] += 1
        x[:, 28] /= 100
        x[:, 29:] = (x[:, 29:] - self.mean[:-1]) / self.std[:-1]
        target = min(self.labels[idx], 1.)
        return x, target


class capacity_Dataset(Dataset):
    training_file = 'train.pkl'
    test_file = 'test.pkl'

    def __init__(self, train=True):
        file_root = '/capacity/'

        data_file = self.training_file if train else self.test_file

        with open(file_root + data_file, "rb") as fp:
            entry = pickle.load(fp)
        self.data = entry['data']
        self.targets_regression = entry['labels_regression']
        self.targets_classification = entry['labels_classification']

        with open(file_root + "standard_deviation.pkl", "rb") as fp:
            mean_std = pickle.load(fp)
            self.mean = mean_std['mean']
            self.std = mean_std['std']

    def __len__(self):
        return len(self.targets_regression)

    def __getitem__(self, idx):
        x = self.data[idx].copy()  # (seq_length, feat_dim) array
        circle = x[:, -14].copy()
        temp = x[:, 51].copy()
        x[:, 6:28] += 1
        x[:, 28] /= 100
        x[:, 29:] = (x[:, 29:] - self.mean[:-1]) / self.std[:-1]
        target = min(self.targets_regression[idx] / 1000., 30.)
        return x, target, circle, temp



class riding_Dataset(Dataset):
    training_file = 'train.pkl'
    test_file = 'test.pkl'

    def __init__(self, train=True):
        file_root = '/riding/'

        data_file = file_root + self.training_file if train else file_root + self.test_file

        with open(data_file, "rb") as fp:
            entry = pickle.load(fp)
        self.data = entry['data']
        self.targets_regression = entry['labels']

        with open(file_root + "standard_deviation.pkl", "rb") as fp:
            entry = pickle.load(fp)
            self.mean = entry["mean"]
            self.std = entry["std"]

    def __len__(self):
        return len(self.targets_regression)

    def __getitem__(self, idx):
        x = self.data[idx].copy()  # (seq_length, feat_dim) array
        circle = x[:, -14].copy()
        temp = x[:, 51].copy()
        date_29 = x[:, 29].copy()
        x[:, 6:28] += 1
        x[:, 28] /= 100
        x[:, 29:] = (x[:, 29:] - self.mean[:-1]) / self.std[:-1]
        target = min(self.targets_regression[idx], 120.)
        return x, target, circle, temp, date_29


if __name__ == '__main__':
    dataset = capacity_Dataset()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)

    x, target, circle, temp = dataset.__getitem__(1)
    print(circle[:10], temp[:10])

