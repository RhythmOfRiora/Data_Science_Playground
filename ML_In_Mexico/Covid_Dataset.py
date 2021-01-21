import pickle
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CovidDataset(Dataset):
    """ML In Mexico Covid dataset."""

    def __init__(self, dataset, get_data=True, transform=None):
        """
        Args:
            dataset (list): Dataset loaded from a pickle file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.full_dataset = dataset
        self.get_data = get_data
        self.transform = transform

    def __len__(self):
        return len(self.full_dataset)  # of how many examples(images?) you have

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        data = self.full_dataset[idx]

        if self.get_data:
            data = self.get_npy_array_from_path(data)

        # Potentially return the labels and the data separately, with extra labesl
        return data


    def get_npy_array_from_path(self, sample):
        #  Returns (npy_array, target)
        sample = (np.load(sample[0]), sample[1])
        # print("sample: ", sample)
        return sample


def load_pickle_file(path):
    try:
        with open(path, 'rb') as file:
            dataset = pickle.load(file)
        return dataset
    except FileNotFoundError as err:
        print("File not found! Please check your input filepath. Full error: ", err)


if __name__ == "__main__":
    training_dataset = load_pickle_file('../train.pckl')
    testing_dataset = load_pickle_file('../testing.pckl')
    validation_dataset = load_pickle_file('../validation.pckl')

    training_covid_dataset = CovidDataset(training_dataset, get_data=True, transform=None)
    training_covid_dataset.__getitem__(2)
    print("Len of training set: ", len(training_covid_dataset))
    print("Training set: ", training_covid_dataset)
    exit()

    testing_covid_dataset = CovidDataset(testing_dataset, get_data=True, transform=None)
    training_covid_dataset.__getitem__(2)
    print("Len of testing set: ", len(testing_covid_dataset))

    validation_covid_dataset = CovidDataset(validation_dataset, get_data=True, transform=None)
    training_covid_dataset.__getitem__(2)
    print("Len of validation set: ", len(validation_covid_dataset))
