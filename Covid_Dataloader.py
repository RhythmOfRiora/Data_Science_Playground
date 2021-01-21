import os
import torch
import random
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader



# class for CovidDataset, inherits from/parent class is Dataloader
class CovidDataset(DataLoader):
    def __init__(self, dataset):
        super(DataLoader, self).__init__()
        self.dataset = pickle.load(open(dataset, "rb"))
        print(f"Now loading data from {dataset}...")
        self.dataset_paths = [x[0] for x in self.dataset]
        self.targets = [x[1] for x in self.dataset]
        self.npy_arrays = self.load_paths(self.dataset_paths)

        # Write a separate class which will be a dataset specific to our needs.
        # Data Loader and dataset class are different things. Instantiate dataset object with path and pass dataset obj
        # into data loader.

    @staticmethod
    def load_paths(paths):
        data = [np.load(path) for path in paths]
        return data

    def get_sample(self, index):
        return tuple(self.npy_arrays[index], self.targets[index])

    def get_random_sample(self):
        index = random.randint(len(self.npy_arrays))
        return tuple(self.npy_arrays[index], self.targets[index])

if __name__ == "__main__":
    training_DL = CovidDataset("/Users/cburn/Data_Science_Playground/train.pckl")
    testing_DL = CovidDataset("/Users/cburn/Data_Science_Playground/testing.pckl")
    validation_DL = CovidDataset("/Users/cburn/Data_Science_Playground/validation.pckl")

# first element in tuple list of file paths. Second element in tuple is list of targets/labels. This means you can load one
# sample at a time. - Lazy loading.








"""Todo: generate labels for the data. Use a tuple (path, label (1, 0) ). The term ‘targets’ means labels. 
Todo: Look at ways to overload and override functions in the dataloaders to do what we need -- make it as generic as possible.  


# there is one dataloader per split.  so we could have 3 dataloaders to “get”.

import Dataloader

def get_training_dataloader(training_set_paths):
	# load the training_set
	training_set = # load the pickle file...e.g. a tuple
	# instantiate your OWN dataloader
	training_dataloader = CovidDataset(training_set)

return training_dataloader



def get_sample(index):
# this loads the actual npy array using the filepath at the specified index 
		x = # data…. np.load(all_npy_arrays[index])
		y = # target or label… targets[index]

	return (x, y) …. x, y

"""