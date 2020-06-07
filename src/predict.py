import os
import time
import warnings

import torch
from dateutil.relativedelta import relativedelta

from dataloader import DataLoader
from model import Model

DEVICE = torch.device('cpu')


def main():
    model = Model().to(DEVICE)
    data_loader = DataLoader(mode='predict')


if __name__ == '__main__':
    os.nice(2)
    torch.set_num_threads(8)
    warnings.filterwarnings('ignore')
    start_time = time.time()

    main()

    time_delta = relativedelta(seconds=(time.time() - start_time))
    print('\n\nTime taken: ' + (' '.join('{} {}'.format(round(getattr(time_delta, k), ndigits=2), k) for k in ['days', 'hours', 'minutes', 'seconds'] if getattr(time_delta, k))))
