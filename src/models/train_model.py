import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import click
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, resize
import torch.optim as optim
import logging
from tempfile import NamedTemporaryFile, TemporaryDirectory
from pathlib import Path
import sys
import pyreadr as py

RANDOM_SEED = 42

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


def get_latest_model_id(dir_name):
    model_ids = list()
    for d in os.listdir(dir_name):
        if os.path.isdir(os.path.join(dir_name, d)):
            try:
                model_ids.append(int(d))
            except ValueError:
                pass
    return max(model_ids) if len(model_ids) else 0


class TEPCNNDataset(Dataset):
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
            .sort_values(by=["faultNumber", "simulationRun", "sample_normalized"], ascending=True) \
            .reset_index(drop=True)

        self.class_count = len(self.df.faultNumber.value_counts())

        self.runs_count = self.df.faultNumber.unique().shape[0] * self.df.simulationRun.unique().shape[0]
        self.sample_count = 960 if is_test else 500
        self.shots_count = self.sample_count - self.window_size + 1

        # making labels according to TEP
        self.labels = self.df.loc[:, ["faultNumber", "sample_normalized"]]
        self.labels.loc[:, "label"] = self.labels.loc[:, "faultNumber"].astype('long')
        if is_test:
            self.labels.loc[(self.labels.label != 0) & (self.labels["sample_normalized"] <= 160), "label"] = 0
        else:
            self.labels.loc[(self.labels["label"] != 0) & (self.labels["sample_normalized"] <= 20), "label"] = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        A sample_normalized means a bunch of measurements of sensors taken at the same time of  a particular simulation run.
        A bunch of such samples compose a "shot" in the following code.
        This code samples a "shot" from a single simulation run.
        A "shot" is a sequence of samples from 52 sensors of len self.window_size.
        The target variable is the value of the "faultNumber" of last item in a "shot" sequence.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx - self.window_size < 0:
            idx = self.window_size

        shot_sample = self.df.iloc[idx, :]["sample_normalized"]

        if shot_sample < self.window_size:
            idx_offset = self.window_size - shot_sample
        else:
            idx_offset = 0

        if idx != self.window_size:
            idx_offset += 1
        shot = self.df.iloc[int(idx - self.window_size + idx_offset):int(idx + idx_offset), :]

        assert shot.iloc[-1]["sample_normalized"] >= self.window_size, "Brah, that's incorrect!"

        shot = shot.iloc[:, 3:].to_numpy()

        label = self.labels.loc[int(idx - idx_offset - 1), "label"]
        label = np.expand_dims(label, axis=[0, 1])
        sample = {'shot': shot, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Net(nn.Module):
    """CNN for TEP dataset"""

    def __init__(self, class_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 40, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_count)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 40)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ToTensor(object):
    def __call__(self, sample):
        shot, label = sample['shot'], sample['label']

        return {'shot': to_tensor(shot).type(torch.FloatTensor),
                'label': to_tensor(label).reshape(-1)}


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        shot, label = sample['shot'], sample['label']
        shot_normed = (shot - TEP_MEAN) / TEP_STD

        return {'shot': shot_normed,
                'label': label}


@click.command()
@click.option('--cuda', required=True, type=int, default=7)
@click.option('-d', '--debug', 'debug', is_flag=True)
def main(cuda, debug):
    logFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.INFO)
    temp_model_dir = TemporaryDirectory(dir="models")
    temp_model_dir.cleanup()
    Path(temp_model_dir.name).mkdir(parents=True, exist_ok=False)
    tempLogFile = os.path.join(temp_model_dir.name, 'log.txt')
    fileHandler = logging.FileHandler(tempLogFile)
    fileHandler.setFormatter(logFormatter)
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(logFormatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    logger.propagate = False

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f'Training begin on {device}')
    loader_jobs = 8
    epochs = 20
    window_size = 30
    batch_size = 64
    tep_file_fault_free_train = "data/raw/TEP_FaultFree_Training.RData"
    tep_file_faulty_train = "data/raw/TEP_Faulty_Training.RData"
    tep_file_fault_free_test = "data/raw/TEP_FaultFree_Testing.RData"
    tep_file_faulty_test = "data/raw/TEP_Faulty_Testing.RData"

    if debug:
        loader_jobs = 0
        batch_size = 2
        tep_file_fault_free_train = "data/raw/sampled_TEP/sampled_train.pkl"
        tep_file_faulty_train = "data/raw/sampled_TEP/sampled_train.pkl"
        tep_file_fault_free_test = "data/raw/sampled_TEP/sampled_test.pkl"
        tep_file_faulty_test = "data/raw/sampled_TEP/sampled_test.pkl"

    transform = transforms.Compose([
        ToTensor(),
        Normalize()
    ])

    trainset = TEPCNNDataset(
        tep_file_fault_free=tep_file_fault_free_train,
        tep_file_faulty=tep_file_faulty_train,
        window_size=window_size,
        is_test=False,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=loader_jobs,
                                              drop_last=True)

    testset = TEPCNNDataset(
        tep_file_fault_free=tep_file_fault_free_test,
        tep_file_faulty=tep_file_faulty_test,
        window_size=window_size,
        is_test=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=loader_jobs,
                                             drop_last=False)

    net = Net(class_count=trainset.class_count).to(device)
    logger.info("\n" + str(net))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if debug:
        epochs = 2
    max_accuracy = 0
    for epoch in range(epochs):

        running_loss = 0.0
        loss_size = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data["shot"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, torch.squeeze(labels))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_size += 1
            log_flag = True if debug else (i + 1) % 2000 == 0
            if log_flag:
                logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / loss_size))
                running_loss = 0.0
                loss_size = 0

            if debug and i > 100:
                break

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data["shot"], data["label"]
                inputs, labels = inputs.to(device), labels.to(device)
                labels = torch.squeeze(labels)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if debug:
                    break
        acc = 100. * correct / total
        logger.info('Epoch %d Accuracy of the network on the %d test images: %d %%' % (epoch, total, acc))
        if acc > max_accuracy:
            logger.info('New best model saved with accuracy %.03f' % acc)
            torch.save(net.state_dict(), os.path.join(temp_model_dir.name, 'latest.pth'))
            max_accuracy = acc

    logger.info(f'Finished Training with max accuracy {round(max_accuracy, 3)}')

    fileHandler.close()

    with open(__file__, 'r') as f:
        with open(os.path.join(temp_model_dir.name, "script.py"), 'w') as out:
            print("# This file was saved automatically during the experiment run.\n", end='', file=out)
            for line in f.readlines():
                print(line, end='', file=out)

    os.rename(
        os.path.join(temp_model_dir.name),
        os.path.join("models", str(get_latest_model_id(dir_name="models") + 1))
    )


if __name__ == '__main__':
    main()
