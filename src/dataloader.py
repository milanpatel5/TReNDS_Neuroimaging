import random
from math import ceil
from threading import Thread

import h5py
import numpy
import pandas
import torch

DATASET_PATH = 'dataset/'


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
            self.data_set_length = len(self.train_set)
            random.shuffle(self.train_set)
        elif mode == 'valid':
            self.predict_set = [int(x) for x in set(self.loading.index.tolist()) - set(self.train_scores.index.tolist())]
            self.length = int(ceil(len(self.predict_set) / self.batch_size))
            self.data_set_length = len(self.predict_set)
        else:
            raise Exception('Please select a valid DataLoader mode...')

        self.idx = 0
        self.batch_data = None
        self.data_preparation_thread = Thread(target=self._prepare_data_())
        self.data_preparation_thread.start()

    def _prepare_data_(self):
        if self.idx == self.data_set_length:
            return

        if self.mode == 'train':
            fmri_batch, loading_batch, target_batch = [], [], []
            batch_cap = self.batch_size
            while batch_cap > 0 and self.idx < self.data_set_length:
                # noinspection PyBroadException
                try:
                    fmri_batch.append(h5py.File(DATASET_PATH + 'fMRI_train/' + str(self.train_set[self.idx]) + '.mat')['SM_feature'].value)
                    loading_batch.append(self.loading.loc[self.train_set[self.idx]].to_numpy())
                    target_batch.append(self.train_scores.loc[self.train_set[self.idx]].to_numpy())
                    self.idx += 1
                    batch_cap -= 1
                except:
                    self.idx += 1
            fmri_batch = torch.tensor(fmri_batch, dtype=torch.float32, device=self.device)
            loading_batch = torch.tensor(loading_batch, dtype=torch.float32, device=self.device)
            target_batch = torch.tensor(target_batch, dtype=torch.float32, device=self.device)
            self.batch_data = (fmri_batch, loading_batch, target_batch)

        elif self.mode == 'predict':
            batch_idx = range(self.idx * self.batch_size, (self.idx + 1) * self.batch_size)

            fmri_batch = [h5py.File(DATASET_PATH + 'fMRI_test/' + str(self.predict_set[idx]) + '.mat')['SM_feature'].value for idx in batch_idx]
            loading_batch = [self.loading.loc[self.predict_set[idx]].to_numpy() for idx in batch_idx]

            fmri_batch = torch.tensor(fmri_batch, dtype=torch.float32, device=self.device)
            loading_batch = torch.tensor(loading_batch, dtype=torch.float32, device=self.device)

        else:
            raise Exception('Please select a valid DataLoader mode...')

    def __next__(self):
        if self.batch_data is None and self.idx == self.data_set_length:
            self._stop_and_reset_idx_()

        self.data_preparation_thread.join()
        data = self.batch_data
        self.batch_data = None
        self.data_preparation_thread = Thread(target=self._prepare_data_())
        self.data_preparation_thread.start()
        return data

    def _stop_and_reset_idx_(self):
        self.idx = 0
        raise StopIteration

    def __len__(self):
        return self.length

    def __iter__(self):
        return self
