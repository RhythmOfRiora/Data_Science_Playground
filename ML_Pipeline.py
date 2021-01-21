import ML_In_Mexico.data_cleaning as data_clean
import ML_In_Mexico.Split_Data as data_splitter
import ML_In_Mexico.config.settings as settings
import ML_In_Mexico.Covid_Dataset as cvd
import pickle
from torch.utils.data import DataLoader

class ML_In_Mexico():

    def __init__(self):
        pass


    def clean_data(self):
        df_covid = data_clean.import_data()
        data_clean.perform_feature_analysis(df_covid)
        data_clean.drop_bad_columns(df_covid)

        print(df_covid.columns)
        columns = settings.columns_of_interest

        cleaned_df = data_clean.remove_non_applicable_data(df_covid)
        features = data_clean.get_feature_vectors(columns, cleaned_df)


    def load_pickle_file(self, path):
        try:
            with open(path, 'rb') as file:
                dataset = pickle.load(file)
            return dataset
        except FileNotFoundError as err:
            print("File not found! Please check your input filepath. Full error: ", err)


    def split_data(self):
        data_splitter.create_dataset_splits(positive_path=settings.positive_path,
                              negative_path=settings.negative_path,
                              train_split=settings.train_split, val_split=settings.val_split,
                              test_split=settings.test_split,
                              save_path=settings.split_data_path,
                              subsample=True)


    def create_datasets(self):
        training_dataset = self.load_pickle_file(f'{settings.split_data_path}/train.pckl')
        testing_dataset = self.load_pickle_file(f'{settings.split_data_path}/testing.pckl')
        validation_dataset = self.load_pickle_file(f'{settings.split_data_path}/validation.pckl')

        training_covid_dataset = cvd.CovidDataset(training_dataset, get_data=True, transform=None)
        training_covid_dataset.__getitem__(2)
        print("Len of training set: ", len(training_covid_dataset))

        testing_covid_dataset = cvd.CovidDataset(testing_dataset, get_data=True, transform=None)
        training_covid_dataset.__getitem__(2)
        print("Len of testing set: ", len(testing_covid_dataset))

        validation_covid_dataset = cvd.CovidDataset(validation_dataset, get_data=True, transform=None)
        training_covid_dataset.__getitem__(2)
        print("Len of validation set: ", len(validation_covid_dataset))

        return testing_covid_dataset, training_covid_dataset, validation_covid_dataset


if __name__ == "__main__":
    pipeline = ML_In_Mexico()
    pipeline.clean_data()
    pipeline.split_data()
    testing_covid_dataset, training_covid_dataset, validation_covid_dataset = pipeline.create_datasets()

    # Define data loader
    # mn_dataset_loader = DataLoader(dataset=training_covid_dataset,
    #                                 batch_size=10,
    #                                 shuffle=False)