import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pyreadr as py
from torchvision.transforms.functional import to_tensor, resize
from random import randint
import gc

# TEP normalisation constants from EDA_v1 notebook
TEP_MEAN = torch.from_numpy(np.array(
    [2.60840858e-01, 3.66377732e+03, 4.50617786e+03, 9.36923827e+00,
     2.69015887e+01, 4.23629281e+01, 2.72214950e+03, 7.48879855e+01,
     1.20400165e+02, 3.45964586e-01, 7.97595706e+01, 4.99911334e+01,
     2.65009318e+03, 2.51278907e+01, 4.99602615e+01, 3.12043499e+03,
     2.29341076e+01, 6.60021453e+01, 2.45631346e+02, 3.40393308e+02,
     9.44453643e+01, 7.70417501e+01, 3.19695759e+01, 8.87928867e+00,
     2.67706384e+01, 6.87406150e+00, 1.87129541e+01, 1.62878283e+00,
     3.26362489e+01, 1.38005826e+01, 2.45676048e+01, 1.25263230e+00,
     1.84738666e+01, 2.22216904e+00, 4.78166224e+00, 2.26500141e+00,
     1.79840572e-02, 8.39498723e-01, 9.74103505e-02, 5.37502095e+01,
     4.37929296e+01, 6.34920553e+01, 5.43011737e+01, 3.01559485e+01,
     6.31554539e+01, 2.29196979e+01, 3.99292803e+01, 3.80739037e+01,
     4.64420345e+01, 5.04799097e+01, 4.19082559e+01, 1.88092347e+01]
)).float()
TEP_STD = torch.from_numpy(np.array(
    [1.46108322e-01, 4.27775994e+01, 1.08699840e+02, 3.56353638e-01,
     2.31067726e-01, 3.13270067e-01, 7.42791897e+01, 1.31549486e+00,
     7.12738852e-02, 8.39785381e-02, 1.75978445e+00, 1.00239752e+00,
     7.48534843e+01, 1.10013251e+00, 1.01898255e+00, 7.70199802e+01,
     6.47620653e-01, 1.81705510e+00, 6.79386884e+01, 1.10078217e+01,
     1.26817415e+00, 1.38709666e+00, 1.73842414e+00, 2.20576788e-01,
     1.92120125e+00, 1.32540424e-01, 9.37602836e-01, 1.24616031e-01,
     2.61142746e+00, 2.85954033e-01, 2.95828205e+00, 1.45611523e-01,
     1.31335193e+00, 1.70181173e-01, 3.40819579e-01, 1.81953767e-01,
     1.01740871e-02, 9.02880601e-02, 1.32383930e-02, 5.81193812e-01,
     6.07383024e-01, 3.26951438e+00, 5.13291828e+00, 2.00389841e+01,
     7.23875632e+00, 1.08172496e+01, 1.26262137e+01, 2.94991280e+00,
     2.35821751e+00, 1.71937564e+01, 9.77333948e+00, 5.06438809e+00]
)).float()


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


class TEPDatasetV4(Dataset):
    """
    This returns the whole len sequences of simulation run.
    'shot': np.array of dim [1, 500, 52]
    'label': np.array of dim [1, 500, 1] or similar.
    Labels are from 0 to 20.
    """

    def __init__(
            self,
            tep_file_fault_free,
            tep_file_faulty,
            # window_size=10,
            is_test=False,
            transform=None,
            for_print=False
    ):
        # self.window_size = window_size
        self.is_test = is_test
        self.transform = transform
        self.for_print = for_print

        if "sampled" in tep_file_fault_free:
            self.df = pd.read_pickle(tep_file_fault_free)
        else:
            fault_free = py.read_r(tep_file_fault_free)
            faulty = py.read_r(tep_file_faulty)
            if is_test:
                self.df = pd.concat([fault_free['fault_free_testing'], faulty['faulty_testing']])
            else:
                self.df = pd.concat([fault_free['fault_free_training'], faulty['faulty_training']])

        # cause the dataset has the broken index
        self.df = self.df \
            .sort_values(by=["faultNumber", "simulationRun", "sample"], ascending=True) \
            .reset_index(drop=True)

        self.class_count = len(self.df.faultNumber.value_counts())

        self.runs_count = self.df.faultNumber.unique().shape[0] * self.df.simulationRun.unique().shape[0]
        self.sample_count = 960 if is_test else 500

        # making labels according to TEP.
        self.labels = self.df.loc[:, ["faultNumber", "simulationRun", "sample"]]
        self.labels.loc[:, "label"] = self.labels.loc[:, "faultNumber"].astype('long')
        if is_test:
            self.labels.loc[(self.labels.label != 0) & (self.labels["sample"] <= 160), "label"] = 0
        else:
            self.labels.loc[(self.labels["label"] != 0) & (self.labels["sample"] <= 20), "label"] = 0

        self.features_count = self.df.shape[1] - 3

        self.raw_data = np.array(
            [g[1].iloc[:, 3:].to_numpy() for g in self.df.groupby(['faultNumber', 'simulationRun'])]
        )
        # for checking if logic above is working properly
        assert np.allclose(
            self.raw_data[0, :, 0],
            self.df[(self.df.simulationRun == 1) & (self.df.faultNumber == 0)].xmeas_1.values
        )
        self.raw_labels = np.array(
            [g[1]["label"].to_numpy() for g in self.labels.groupby(['faultNumber', 'simulationRun'])]
        )

        self.max_sim_run_number = int(self.df.simulationRun.max())
        self.print_sim_run = randint(0, self.max_sim_run_number - 1)
        """
        This is for selecting the proper indices in row_data so the data set contains the only batch 
        of len self.class_count where each fault type occurs exactly once.
        """
        self.print_ids = [ft * self.max_sim_run_number + self.print_sim_run for ft in range(self.class_count)]

        del self.df
        gc.collect()

        permutation = np.random.permutation(self.raw_data.shape[0])
        if not self.for_print:
            self.raw_data = self.raw_data[permutation]
            self.raw_labels = self.raw_labels[permutation]


    def __len__(self):
        if self.for_print:
            return len(self.raw_data[self.print_ids, ...])
        else:
            return len(self.raw_data)

    def __getitem__(self, idx):
        """
        Just sample from self.raw_data and self.raw_labels according to the idx.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.for_print:
            shot_sample = self.raw_data[self.print_ids, ...][idx, ...]
            label_sample = self.raw_labels[self.print_ids, ...][idx, ...]
        else:
            shot_sample = self.raw_data[idx, ...]
            label_sample = self.raw_labels[idx, ...]

        label_sample = np.expand_dims(label_sample, axis=[1])
        sample = {'shot': shot_sample, 'label': label_sample}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def change_print_sim_run(self):
        self.print_sim_run = randint(0, self.max_sim_run_number - 1)
        self.print_ids = [ft * self.max_sim_run_number + self.print_sim_run for ft in range(self.class_count)]

    # def shuffle(self):
    #     self.permutation = np.random.permutation(self.raw_data.shape[0])


class TEPRNNGANDataset(Dataset):
    def __init__(
            self,
            tep_file_fault_free,
            tep_file_faulty,
            window_size=10,
            is_test=False,
            transform=None
    ):
        self.window_size = window_size
        self.is_test = is_test
        self.transform = transform

        if "sampled" in tep_file_fault_free:
            self.df = pd.read_pickle(tep_file_fault_free)
        else:
            fault_free = py.read_r(tep_file_fault_free)
            faulty = py.read_r(tep_file_faulty)
            if is_test:
                self.df = pd.concat([fault_free['fault_free_testing'], faulty['faulty_testing']])
            else:
                self.df = pd.concat([fault_free['fault_free_training'], faulty['faulty_training']])

        # cause the dataset has the broken index
        self.df = self.df \
            .sort_values(by=["faultNumber", "simulationRun", "sample"], ascending=True) \
            .reset_index(drop=True)

        self.class_count = len(self.df.faultNumber.value_counts())

        self.runs_count = self.df.faultNumber.unique().shape[0] * self.df.simulationRun.unique().shape[0]
        self.sample_count = 960 if is_test else 500
        self.shots_count = self.sample_count - self.window_size + 1

        # making labels according to TEP
        self.labels = self.df.loc[:, ["faultNumber", "sample"]]
        self.labels.loc[:, "label"] = self.labels.loc[:, "faultNumber"].astype('long')
        if is_test:
            self.labels.loc[(self.labels.label != 0) & (self.labels["sample"] <= 160), "label"] = 0
        else:
            self.labels.loc[(self.labels["label"] != 0) & (self.labels["sample"] <= 20), "label"] = 0

        self.features_count = self.df.shape[1] - 3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        A sample means a bunch of measurements of sensors taken at the same time of  a particular simulation run.
        A bunch of such samples compose a "shot" in the following code.
        This code samples a "shot" from a single simulation run.
        A "shot" is a sequence of samples from 52 sensors of len self.window_size.
        The target variable is the value of the "faultNumber" of last item in a "shot" sequence.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx - self.window_size < 0:
            idx = self.window_size

        shot_sample = self.df.iloc[idx, :]["sample"]

        if shot_sample < self.window_size:
            idx_offset = self.window_size - shot_sample
        else:
            idx_offset = 0

        if idx != self.window_size:
            idx_offset += 1
        shot = self.df.iloc[int(idx - self.window_size + idx_offset):int(idx + idx_offset), :]

        assert shot.iloc[-1]["sample"] >= self.window_size, "Brah, that's incorrect!"

        shot = shot.iloc[:, 3:].to_numpy()

        label = self.labels.iloc[int(idx - self.window_size + idx_offset):int(idx + idx_offset), :]["label"].to_numpy()
        label = np.expand_dims(label, axis=[1])
        sample = {'shot': shot, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        shot, label = sample['shot'], sample['label']

        return {'shot': to_tensor(shot).type(torch.FloatTensor),
                'label': to_tensor(label)}


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        shot, label = sample['shot'], sample['label']
        shot_normed = (shot - TEP_MEAN) / TEP_STD

        return {'shot': shot_normed,
                'label': label}


class InverseNormalize(object):
    def __init__(self):
        pass

    def __call__(self, sample_normalized):
        shot, label = sample_normalized['shot'], sample_normalized['label']
        shot_original = shot * TEP_STD + TEP_MEAN

        return {'shot': shot_original,
                'label': label}

