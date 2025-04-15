import os
import torch
from torchvision.datasets import *
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from .utils import create_centerlized_datasets, annotate_data

# PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
# BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class DataModule(pl.LightningDataModule):
    def __init__(self, data_config, train_data, test_data, num_classes= 10, batch_size=16, num_workers= 36):
        super().__init__()

        self.data_config = data_config
        
        self.train_data = train_data
        self.test_data = test_data

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.num_workers = num_workers
        self.num_classes = num_classes
        self.batch_size = batch_size

    def prepare_data(self):
        # download
        MNIST('./data/', train=True, download=True)
        MNIST('./data/', train=False, download=True)

    def setup(self, stage=None):
        #train_dataset, truth_dataset, test_dataset = create_centerlized_datasets(data_config=self.data_config)
        #train_data = annotate_data(data=train_dataset, truth_data=truth_dataset, num_clusters=self.num_clusters)
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self._train, self._val = self.train_data, self.test_data # random_split(self.train_data, [9000, 1000])
            #print(f'{self._train}')
            #mnist_full = MNIST('./data/', train=True, transform=self.transform)
            #self._train, self._val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self._test = self.test_data
            #self._test = MNIST('./data/', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self._train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self._val, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self._test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers
        )