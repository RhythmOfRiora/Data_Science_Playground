import pandas as pd
import copy
import random

binary_columns = ["icu", "covid_res", "contact_other_covid", "tobacco",
                  "renal_chronic", "obesity", "cardiovascular",
                  "other_disease", "hypertension", "inmsupr", "asthma",
                  "copd", "diabetes", "pregnancy", "pneumonia", "intubed",
                  "patient_type", "sex"]

def import_data():
    df_covid = pd.read_csv("/Users/cburn/Downloads/archive/covid.csv")
    # print(df_covid)
    return df_covid


def remove_non_applicable_data(df_covid):
    valid_values = [1, 2]
    df_random = copy.deepcopy(df_covid)
    df_biased_to_no = copy.deepcopy(df_covid)


    # This Dataframe drops any rows that don't have values 1 or 2 - and so, will not be very large.
    for col in binary_columns:
        df_covid.drop(df_covid[df_covid[col] > 2].index, inplace=True)


    # This Dataframe randomly assigns 1 or 2 to rows which don't have values 1 or 2 in a column.
    for col in binary_columns:
        df_random.loc[df_random[col] > 2, col] = random.choice([1, 2])

    # This Dataframe assumes that if an answer is not "Yes" to having a certain condition, then it is "No" for every
    # column i.e. the pregnancy column.
    for col in binary_columns:
        df_biased_to_no.loc[df_biased_to_no[col] > 1, col] = 2


    return df_covid, df_random, df_biased_to_no


def clean_binary_columns(fully_cleaned_df_covid, df_random, df_biased_to_no):
    df_list = [fully_cleaned_df_covid, df_random, df_biased_to_no]

    for dataframe in df_list:
        for col in binary_columns:
            dataframe.loc[dataframe[col] > 1, col] = 0


if __name__ == "__main__":
    df_covid = import_data()
    cleaned_df, df_rand, df_biased = remove_non_applicable_data(df_covid)
    clean_binary_columns(cleaned_df, df_rand, df_biased)
    print(df_rand)