import os
import math
import ML_In_Mexico.config.settings as settings
import pickle
from random import sample
from random import choice


def create_dataset_splits(positive_path, negative_path, train_split=0.8, val_split=0.1, test_split=0.1, save_path="/tmp/ml_data/", subsample=True):
    """
    intubed_path: path to the folder containing all the intubated samples that are .npy arrays.
    not_tubed_path: path to the folder containing all the not intubated samples that are .npy arrays.
    train_split: the fraction to use for training
    val_split: the fraction to use for validation
    test_split: the fraction to use for testing
    save_path: where to save the splits
    subsample: means whether or not to subsample!
    """


    train_split = 0.8 # 80%
    val_split = 0.1 # 10%
    test_split = 0.1 # 10%

    # do some validation on the split!  make sure they add up to 1, and that they are all 0<=x<=1

    save_path = "/Users/cburn/Data_Science_Playground/ML_In_Mexico/split_data_dir"

    # enumerate all the intubated samples
    intubated_samples = os.scandir(positive_path)
    non_intubated_samples = os.scandir(negative_path)

    # enumerate all the not_intubed samples

    # use os.walk, scandir

    intubed_samples = [(str(entry.path), 1) for entry in intubated_samples] # 2560, positive case
    print(f"Number of intubed sample files: {len(intubed_samples)}")

    not_intubed_samples = [(str(entry.path), 0) for entry in non_intubated_samples] #20600, negative case
    print(f"Number of non-intubed sample files: {len(not_intubed_samples)}")

    # positive cases should always be in the minority in binary classification problems.
    # check if we want to subsample
    if subsample:
    	# do subsampling!
    	# find the bigger class
    	# if it is intubed samples (unlikely!) then subsample these to get (not_intubed_samples - 1intubed+1) intubated samples
    	# if is not_intubated_samples (very likely!) then subsample to get (intubed_samples+1) not_intubed samples
    	# use the random library for example….should be able to pass in a list, and the number you want to subsample….e.g. random.subsample(not_intubed_samples, 2561)
    # 2561 intubated, 2562 not intubated

        # Don't generalize - distinguish between the positive dataset and the negative one.
        num_samples_needed = min(len(intubed_samples), len(not_intubed_samples)) + 1

        print(f"Number of samples needed: {num_samples_needed}")
        random_sample_list = sample(not_intubed_samples, num_samples_needed)

        print(f"Length of random sample list: {len(random_sample_list)}")
        # do the splits! using indices….
    #
        num_train_intubated = math.floor(train_split * len(intubed_samples)) # might need math.floor...prob will to get an int back
        num_val_intubated = math.floor(val_split * len(intubed_samples)) # might need math.floor...prob will to get an int back
        num_test_intubated = len(intubed_samples) - num_val_intubated - num_train_intubated

        num_train_not_intubated = math.floor(train_split * len(random_sample_list)) # might need math.floor...prob will to get an int back
        num_val_not_intubated = math.floor(val_split * len(random_sample_list)) # might need math.floor...prob will to get an int back
        num_test_not_intubated = len(random_sample_list) - num_val_not_intubated - num_train_not_intubated
    #
    # # then repeat that (above) for the not_intubated
    #
    # # again there are subsampling functions….these will return a random n indices from a list, and I think you can specify indices to exclude...maybe
    #
        train_split_intubated_filepaths = sample(intubed_samples, num_train_intubated)

        val_split_intubated_filepaths = custom_sample(intubed_samples, num_needed=num_val_intubated,
                                                    exclude=train_split_intubated_filepaths)


        test_split_intubated_filepaths = custom_sample(intubed_samples, num_needed=num_test_intubated,
                                                     exclude=train_split_intubated_filepaths + val_split_intubated_filepaths)


    #
    # # now we have the indices, great... go get the actual files.
    #
    # train_split_intubated_filepaths =  [intubed_samples[index] for index in train_split_intubated_indices]
    # do the same for train/val/test for intubated and not intubated

        train_split_not_intubated_filepaths = sample(random_sample_list, num_train_not_intubated)
        val_split_not_intubated_filepaths = custom_sample(random_sample_list, num_needed=num_val_not_intubated,
                                                        exclude=train_split_not_intubated_filepaths)

        test_split_not_intubated_filepaths = custom_sample(random_sample_list, num_needed=num_test_not_intubated,
                                                         exclude=train_split_not_intubated_filepaths + val_split_not_intubated_filepaths)

         # add the train_intubated + train_not_intubated together etc etc etc
        print(len(train_split_intubated_filepaths), len(train_split_not_intubated_filepaths))

        training_data = train_split_intubated_filepaths + train_split_not_intubated_filepaths
        print(len(training_data))
        validation_data = val_split_intubated_filepaths + val_split_not_intubated_filepaths
        testing_data = test_split_intubated_filepaths + test_split_not_intubated_filepaths


    # do some more validation….so check nothing in the lists for the file_paths overlaps, and that the total num of elements = the original amount, good idea to print to screen
        print(f"num_train_intubated : {num_train_intubated}")
        print(f"num_val_intubated: {num_val_intubated}")
        print(f"num_test_intubated : {num_test_intubated}")

        print(f"num_train_not_intubated : {num_train_not_intubated}")
        print(f"num_val_not_intubated: {num_val_not_intubated}")
        print(f"num_test_not_intubated : {num_test_not_intubated}")

        print(f"training data length: {len(training_data)} (should be {num_train_intubated} + {num_train_not_intubated} = {num_train_intubated + num_train_not_intubated})")
        print(f"validation data length: {len(validation_data)} (should be {num_val_intubated} + {num_val_not_intubated} = {num_val_intubated + num_val_not_intubated})")
        print(f"testing data length: {len(testing_data)} (should be {num_test_intubated} + {num_test_not_intubated} = {num_test_intubated + num_test_not_intubated})")

        # # then save each split as a pickle file to disk
        #
        pickle.dump(training_data, open(f"{save_path}/train_balanced.pckl", "wb"))
        pickle.dump(validation_data, open(f"{save_path}/validation_balanced.pckl", "wb"))
        pickle.dump(testing_data, open(f"{save_path}/testing_balanced.pckl", "wb"))


def custom_sample(data, num_needed, exclude=None):
    # Add comments here
    choices = set(data) - set(exclude)
    if len(list(choices)) >= num_needed:
        return sample(list(choices), num_needed)
    else:
        raise Exception("You're asking for more samples than you have unique data records.")



if __name__ == "__main__":
    create_dataset_splits(positive_path=settings.positive_path,
                          negative_path=settings.negative_path,
                          train_split=0.8, val_split=0.1, test_split=0.1, \
                          save_path="/Users/cburn/Data_Science_Playground/ML_In_Mexico/split_data_dir",
                          subsample=True)

    # encapsulate descriptions in file names -- parameterize the save file names & paths.
    # Look at data loaders!

