from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
import torch
import numpy as np


class MNIST_DataModule(pl.LightningDataModule):
    '''
        DataModule to hold the MNIST dataset. Accepts different transforms for train and test to
        allow for extrapolation experiments.

        Parameters
        ----------
        data_dir : str
            Directory where MNIST will be downloaded or taken from.

        train_transform : [transform] 
            List of transformations for the training dataset. The same
            transformations are also applied to the validation dataset.

        test_transform : [transform] or [[transform]]
            List of transformations for the test dataset. Also accepts a list of
            lists to validate on multiple datasets with different transforms.

        batch_size : int
            Batch size for both all dataloaders.
    '''

    def __init__(self, data_dir='./', train_transform=None, test_transform=None, batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data (train/val and test sets)
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''
            Called on each GPU separately - stage defines if we are
            at fit, validate, test or predict step.
        '''
        # we set up only relevant datasets when stage is specified
        if stage in [None, 'fit', 'validate']:
            mnist_train = MNIST(self.data_dir, train=True,
                                transform=(self.train_transform or self.default_transform))
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == 'test' or stage is None:
            if self.test_transform is None or isinstance(self.test_transform, transforms.Compose):
                self.mnist_test = MNIST(self.data_dir,
                                        train=False,
                                        transform=(self.test_transform or self.default_transform))
            else:
                self.mnist_test = [MNIST(self.data_dir,
                                         train=False,
                                         transform=test_transform)
                                   for test_transform in self.test_transform]

    def train_dataloader(self):
        '''returns training dataloader'''
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)
        return mnist_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        '''returns test dataloader(s)'''
        if isinstance(self.mnist_test, MNIST):
            return DataLoader(self.mnist_test, batch_size=self.batch_size)

        mnist_test = [DataLoader(test_dataset,
                                 batch_size=self.batch_size)
                      for test_dataset in self.mnist_test]
        return mnist_test


def get_transforms():
    # define transformations
    _transform = lambda degrees: transforms.Compose([
        transforms.RandomRotation(degrees=degrees),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    degrees = np.linspace(0, 180, 7)
    test_transform = np.vectorize(_transform)(degrees)

    train_transform = test_transform[0]

    return train_transform, test_transform

def get_transforms2():
    # define transformations
    _transform = lambda degrees: transforms.Compose([
        transforms.RandomRotation(degrees=[degrees]*2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    degrees = np.linspace(0, 180, 7)
    test_transform = np.vectorize(_transform)(degrees)

    train_transform = test_transform[0]

    return train_transform, test_transform


def get_occlusion_transforms():
    # define transformations
    _transform = lambda p: transforms.Compose([
        transforms.PILToTensor(),
        # transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=p, value='random'),
    ])

    ps = np.linspace(0, 1, 11)
    test_transform = np.vectorize(_transform)(ps)

    train_transform = test_transform[0]


    return train_transform, test_transform


def get_pixelated_transforms():
    # define transformations
    _transform = lambda p: transforms.Compose([
        transforms.PILToTensor(),
        # transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(p),
        transforms.Resize(28),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    ps = np.arange(28, 1, step=-4)
    test_transform = [_transform(int(p)) for p in ps]

    train_transform = test_transform[0]


    return train_transform, test_transform


def get_perspective_transforms():
    # define transformations
    _transform = lambda p: transforms.Compose([
        transforms.PILToTensor(),
        # transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomPerspective(p, p=1),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    ps = np.linspace(0, 1, 11)
    # ps = rots
    test_transform = np.vectorize(_transform)(ps)

    train_transform = test_transform[0]


    return train_transform, test_transform