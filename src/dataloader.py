import random
from math import ceil

import h5py
import numpy
import pandas
import torch

DATASET_PATH = '../../input/trends-assessment-prediction/'


class DataLoader:
    def __init__(self, mode, device, batch_size=1):
        self.device = device
        self.batch_size = batch_size
        self.mode = mode

        self.loading = pandas.read_csv(DATASET_PATH + 'loading.csv', index_col='Id', dtype=numpy.float32)
        self.train_scores = pandas.read_csv(DATASET_PATH + 'train_scores.csv', index_col='Id', dtype=numpy.float32)
        self.train_scores = self.train_scores.fillna(self.train_scores.mean())

        if mode == 'train':
            self.train_set = [int(x) for x in self.train_scores.index.tolist()]
            self.length = int(ceil(len(self.train_set) / self.batch_size))
        elif mode == 'valid':
            self.predict_set = [int(x) for x in set(self.loading.index.tolist()) - set(self.train_scores.index.tolist())]
            self.length = int(ceil(len(self.predict_set) / self.batch_size))
        else:
            raise Exception('Please select a valid DataLoader mode...')

        self.idx = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        if self.idx == 0 and self.mode == 'train':
            random.shuffle(self.train_set)
        elif self.idx == self.length:
            self._stop_and_reset_idx_()

        if self.mode == 'train':
            fmri_batch, loading_batch, target_batch = [], [], []
            batch_idx = range(self.idx * self.batch_size, (self.idx + 1) * self.batch_size)
            if len(batch_idx) <= 1:
                self._stop_and_reset_idx_()
            for idx in batch_idx:
                fmri_batch.append(h5py.File(DATASET_PATH + 'fMRI_train/' + str(self.train_set[idx]) + '.mat')['SM_feature'].value)
                loading_batch.append(self.loading.loc[self.train_set[idx]].to_numpy())
                target_batch.append(self.train_scores.loc[self.train_set[idx]].to_numpy())
            fmri_batch = torch.tensor(fmri_batch, dtype=torch.float32, device=self.device)
            loading_batch = torch.tensor(loading_batch, dtype=torch.float32, device=self.device)
            target_batch = torch.tensor(target_batch, dtype=torch.float32, device=self.device)
            return fmri_batch, loading_batch, target_batch

        elif self.mode == 'predict':
            fmri_batch, loading_batch = [], []
            for idx in range(self.idx * self.batch_size, (self.idx + 1) * self.batch_size):
                fmri_batch.append(h5py.File(DATASET_PATH + 'fMRI_test/' + str(self.predict_set[idx]) + '.mat')['SM_feature'].value)
                loading_batch.append(self.loading.loc[self.predict_set[idx]].to_numpy())
            fmri_batch = torch.tensor(fmri_batch, dtype=torch.float32, device=self.device)
            loading_batch = torch.tensor(loading_batch, dtype=torch.float32, device=self.device)
            return fmri_batch, loading_batch

        else:
            raise Exception('Please select a valid DataLoader mode...')

    def __len__(self):
        return self.length

    def _stop_and_reset_idx_(self):
        self.idx = -1
        raise StopIteration
