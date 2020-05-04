import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import click
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, normalize, resize
import torch.optim as optim
import logging
from tempfile import NamedTemporaryFile, TemporaryDirectory
from pathlib import Path
import sys
from PIL import Image
from typing import Iterable

RANDOM_SEED = 42


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
            tep_file,
            window_size=10,
            is_test=False,
            transform=None
    ):
        self.tep_file = tep_file
        self.window_size = window_size
        self.is_test = is_test
        self.transform = transform

        self.df = pd.read_pickle(tep_file)
        # cause the dataset has the broken index
        self.df = self.df\
            .sort_values(by=["faultNumber", "simulationRun", "sample"], ascending=True)\
            .reset_index(drop=True)

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        This samples a "shot" from a single simulation run.
        A shot is a sequence of samples from 52 sensors of len self.window_size.
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

        label = self.labels.loc[int(idx - idx_offset - 1), "label"]
        label = np.expand_dims(label, axis=[0, 1])
        sample = {'shot': shot, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 40, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 40)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': resize(image, self.size, interpolation=self.interpolation),
                'label': label}


class ToTensor(object):
    def __call__(self, sample):
        shot, label = sample['shot'], sample['label']

        return {'shot': to_tensor(shot).type(torch.FloatTensor),
                'label': to_tensor(label).reshape(-1)}


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': normalize(image, self.mean, self.std, self.inplace),
                'label': label}


@click.command()
@click.option('-d', '--debug', 'debug', is_flag=True)
def main(debug):
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
    window_size = 30

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Training begin on {device}')
    loader_jobs = 8
    epochs = 20
    if debug:
        loader_jobs = 0

    transform = transforms.Compose([
        ToTensor(),
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = TEPCNNDataset(
        tep_file="data/raw/sampled_TEP/sampled_train.pkl",
        window_size=window_size,
        is_test=False,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=loader_jobs,
                                              drop_last=True)

    testset = TEPCNNDataset(
        tep_file="data/raw/sampled_TEP/sampled_test.pkl",
        window_size=window_size,
        is_test=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=loader_jobs,
                                             drop_last=False)

    net = Net().to(device)

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

            # if debug:
            #     break

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
    os.rename(
        os.path.join(temp_model_dir.name),
        os.path.join("models", str(get_latest_model_id(dir_name="models") + 1))
    )


if __name__ == '__main__':
    main()
