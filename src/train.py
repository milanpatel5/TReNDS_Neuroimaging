import os
import time
import warnings

import torch
from dateutil.relativedelta import relativedelta
from torch.nn import Module, L1Loss
from torch.optim.adam import Adam
from tqdm import tqdm

from dataloader import DataLoader
from model import Model

DEVICE = torch.device('cpu')
N_EPOCH = 2


def main():
    # noinspection PyUnresolvedReferences
    model = Model().to(DEVICE)
    optimizer = Adam(params=model.parameters(), lr=2e-3, weight_decay=1e-3, amsgrad=True)
    loss_fn = FeatureWeightedLoss()
    data_loader = DataLoader(mode='train', batch_size=2, device=DEVICE)

    for epoch_idx in range(0, N_EPOCH):
        progress_bar = tqdm(data_loader)
        mean_loss = 0
        for itr, (fmri, loading, target) in enumerate(progress_bar):
            progress_bar.set_description_str('Training epoch: {}/{}'.format(epoch_idx + 1, N_EPOCH))
            optimizer.zero_grad()
            output = model(fmri, loading)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            mean_loss = (mean_loss * itr * 1e-2 + loss.item()) / (itr * 1e-2 + 1)
            progress_bar.set_postfix_str('loss={:.4f}'.format(mean_loss))
            if itr % 10 == 0:
                model.save()


class FeatureWeightedLoss(Module):
    def __init__(self):
        super().__init__()
        self.w = torch.tensor([0.3, 0.175, 0.175, 0.175, 0.175], dtype=torch.float32, device=DEVICE)
        self.l1_loss = L1Loss(reduction='none')

    def forward(self, output, target):
        loss = self.l1_loss(output, target)
        loss = torch.sum(self.w * (torch.sum(loss, dim=0) / torch.sum(target, dim=0)))
        return loss


if __name__ == '__main__':
    os.nice(2)
    torch.set_num_threads(8)
    warnings.filterwarnings('ignore')
    start_time = time.time()

    main()

    time_delta = relativedelta(seconds=(time.time() - start_time))
    print('\n\nTime taken: ' + (' '.join('{} {}'.format(round(getattr(time_delta, k), ndigits=2), k) for k in ['days', 'hours', 'minutes', 'seconds'] if getattr(time_delta, k))))
