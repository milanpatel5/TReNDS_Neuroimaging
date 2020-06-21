import time
import warnings

import torch
from dateutil.relativedelta import relativedelta
from torch.nn import Module, L1Loss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from dataloader import DataLoader
from model import Model

DEVICE = torch.device('cuda')
N_EPOCH = 50


# noinspection PyTypeChecker
# noinspection PyUnresolvedReferences
def main():
    model = Model().to(DEVICE)
    model.load()

    optimizer = Adam(params=model.parameters(), lr=0.001)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, eta_min=0.00001)
    loss_fn = FeatureWeightedLoss()

    model.train()
    for epoch_idx in range(10, N_EPOCH):
        with tqdm(DataLoader(mode='train', batch_size=4, device=DEVICE)) as progress_bar:
            mean_loss = 0
            for itr, (fmri, loading, target) in enumerate(progress_bar):
                if fmri.shape[0] > 1:
                    progress_bar.set_description_str('Training epoch: {}/{}'.format(epoch_idx + 1, N_EPOCH))
                    optimizer.zero_grad()
                    output = model(fmri, loading)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()

                    mean_loss = (mean_loss * itr + loss.item()) / (itr + 1)
                    progress_bar.set_postfix_str('lr={:.4f}, loss={:.4f}'.format(lr_scheduler.get_last_lr()[0], mean_loss))
                lr_scheduler.step(epoch_idx + itr / len(progress_bar))
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
    # os.nice(2)
    torch.set_num_threads(8)
    warnings.filterwarnings('ignore')
    start_time = time.time()

    main()

    time_delta = relativedelta(seconds=int(time.time() - start_time))
    print('\n\nTime taken: ' + (' '.join('{} {}'.format(round(getattr(time_delta, k), ndigits=2), k) for k in ['days', 'hours', 'minutes', 'seconds'] if getattr(time_delta, k))))
