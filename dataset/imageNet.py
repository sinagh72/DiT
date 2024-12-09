"""
imageNet.py: ImageNet script
__author: Sina Gholami
__update: created
__update_date: 11/15/2024
__note: working with python <= 3.10
"""
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
from PIL import Image
from huggingface_hub import login
from datasets import load_dataset
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torch

class ImageNetDataset(Dataset):

    def __init__(self, transform=None, dataset=None):
        """
        hugging face login
        """
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, index):
        return self.transform(self.dataset[index]["image"]), (self.dataset[index]["label"], 0)

    def __len__(self):
        return len(self.dataset)


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=None, train_transform=None, val_transform=None):
        """
        :param batch_size: int
        :param train_transform: transforms
        :param test_transform: transforms
        """
        login(token="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")
       
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        # Assign Train for use in Dataloaders
        if stage == "train":
            train_data = load_dataset("imagenet-1k", split="train", trust_remote_code=True)
            self.data_train = ImageNetDataset(transform=self.train_transform, dataset=train_data)
            print("ImageNet train data len:", len(self.data_train))
        # Assign val split(s) for use in Dataloaders
        elif stage == "val":
            val_data = load_dataset("imagenet-1k", split="val", trust_remote_code=True)
            self.data_val = ImageNetDataset(transform=self.val_transform, dataset=val_data)
            print("ImageNet val data len:", len(self.data_val))

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            test_data = load_dataset("imagenet-1k", split="test", trust_remote_code=True)
            self.data_test = ImageNetDataset(transform=self.val_transform, dataset=test_data)
            print("ImageNet test data len:", len(self.data_test))

    def train_dataloader(self, shuffle: bool = True, drop_last: bool = True, pin_memory: bool = True,
                         workers: int = torch.cuda.device_count() * 2):
        """
        :param num_workers: int, number of workers for training loder training
        """
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          pin_memory=pin_memory,
                          drop_last=drop_last,
                          num_workers=workers)

    def val_dataloader(self, shuffle: bool = False, drop_last: bool = False, pin_memory: bool = True,
                       workers: int = torch.cuda.device_count() * 2):
        return DataLoader(self.data_val,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          num_workers=workers,
                          pin_memory=pin_memory)

    def test_dataloader(self, shuffle: bool = False, drop_last: bool = False, pin_memory: bool = True,
                        workers: int = torch.cuda.device_count() * 2):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          num_workers=workers,
                          pin_memory=pin_memory)
    