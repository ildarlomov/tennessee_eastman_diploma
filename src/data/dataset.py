import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pyreadr as py

class TEPDataset(Dataset):
    """Btp time series dataset."""

    def __init__(self, tep_file_fault_free, tep_file_faulty, is_test=False, normalize=True):
        """
        Args:
            csv_file (string): path to csv file
            normalize (bool): whether to normalize the data in [-1,1]
        """
        if "sampled" in tep_file_fault_free:
            df = pd.read_pickle(tep_file_fault_free)
        else:
            fault_free = py.read_r(tep_file_fault_free)
            faulty = py.read_r(tep_file_faulty)
            if is_test:
                df = pd.concat([fault_free['fault_free_testing'], faulty['faulty_testing']])
            else:
                df = pd.concat([fault_free['fault_free_training'], faulty['faulty_training']])

        # todo: add conditioning on fault number, now we generate only the normal condition
        df = df[(df.faultNumber == 0)]
        work_with_columns = ['faultNumber', 'simulationRun', 'sample', 'xmeas_1']
        raw_data = torch.from_numpy(
            np.expand_dims(
                np.array(
                    [g[1]["xmeas_1"] for g in df[work_with_columns].groupby(['faultNumber', 'simulationRun'])]
                ), -1)
        ).float()
        # for checking if logic above is working properly
        assert np.allclose(raw_data.squeeze()[0, :].numpy(), df[(df.simulationRun == 1) & (df.faultNumber == 0)].xmeas_1.values)
        self.data = self.normalize(raw_data) if normalize else raw_data
        self.seq_len = raw_data.size(1)

        # Estimates distribution parameters of deltas (Gaussian) from normalized data
        original_deltas = raw_data[:, -1] - raw_data[:, 0]
        self.original_deltas = original_deltas
        self.or_delta_max, self.or_delta_min = original_deltas.max(), original_deltas.min()
        deltas = self.data[:, -1] - self.data[:, 0]
        self.deltas = deltas
        self.delta_mean, self.delta_std = deltas.mean(), deltas.std()
        self.delta_max, self.delta_min = deltas.max(), deltas.min()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min()) / (x.max() - x.min()) - 1)

    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x * self.max - x * self.min + self.max + self.min)

    def sample_deltas(self, number):
        """Sample a vector of (number) deltas from the fitted Gaussian"""
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std

    def normalize_deltas(self, x):
        return ((self.delta_max - self.delta_min) * (x - self.or_delta_min) / (
                    self.or_delta_max - self.or_delta_min) + self.delta_min)

