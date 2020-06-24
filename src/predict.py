import csv
import time
import warnings

import torch
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from dataloader import DataLoader
from model import Model

DEVICE = torch.device('cuda')


def main():
    model = Model()
    model.load(device=DEVICE)

    with open('predictions.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Id', 'Predicted'])
        attrs = ['_age', '_domain1_var1', '_domain1_var2', '_domain2_var1', '_domain2_var2']

        with tqdm(DataLoader(mode='predict', batch_size=2, device=DEVICE)) as progress_bar:
            progress_bar.set_description_str('Generating predictions')
            for itr, (ids, fmri, loading) in enumerate(progress_bar):
                output = model(fmri, loading)
                output = output.detach().cpu().numpy()
                for prediction_id, predicted_values in zip(ids, output):
                    for attr, predicted_value in zip(attrs, predicted_values):
                        csv_writer.writerow([str(int(prediction_id)) + attr, predicted_value])
        csv_file.close()


if __name__ == '__main__':
    # os.nice(2)
    torch.set_num_threads(12)
    warnings.filterwarnings('ignore')
    start_time = time.time()

    main()

    time_delta = relativedelta(seconds=int(time.time() - start_time))
    print('\n\nTime taken: ' + (' '.join('{} {}'.format(round(getattr(time_delta, k), ndigits=2), k) for k in ['days', 'hours', 'minutes', 'seconds'] if getattr(time_delta, k))))
