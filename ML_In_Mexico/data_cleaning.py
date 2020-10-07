import pandas as pd
import copy
import random

binary_columns = ["icu", "covid_res", "contact_other_covid", "tobacco",
                  "renal_chronic", "obesity", "cardiovascular",
                  "other_disease", "hypertension", "inmsupr", "asthma",
                  "copd", "diabetes", "pregnancy", "pneumonia", "intubed",
                  "patient_type", "sex"]

def import_data():
    df_covid = pd.read_csv("/Users/cburn/Downloads/archive/covid.csv", parse_dates=['date_symptoms',
                                                                                    'date_died', 'entry_date'])
    # print(df_covid)
    return df_covid


def feature_analysis(df_covid):
    with open("/tmp/covid_dataset_analysis.txt", "w") as f:
        for index, column in enumerate(list(df_covid)):
            unique_values = df_covid[column].unique()

            if "id" in column:
                unique_ids = df_covid[column].nunique()
                df_row_count = len(df_covid)

                if df_row_count == unique_ids:
                    print(f"Number of Unique IDs ({unique_ids}) is equal to Number of Rows in Dataframe ({df_row_count}), all good.")
                    f.write(f"\n Number of Unique IDs ({unique_ids}) is equal to Number of Rows in Dataframe ({df_row_count}), all good.\n")
                else:
                    print(f"Number of Unique IDs ({unique_ids}) is not equal to Number of Rows in Dataframe ({df_row_count}), please check this.")
                    f.write(f"\n Number of Unique IDs ({unique_ids}) is not equal to Number of Rows in Dataframe ({df_row_count}), please check this. \n")

            if "date" in column:
                print(f"Earliest Date In Column {column}: ", df_covid[column].min())
                f.write(f"\nEarliest Date In Column {column}: " + str(df_covid[column].min()) + "\n")

                print(f"Latest Date In Column {column}: ", df_covid[column].max())
                f.write(f"\nLatest Date In Column {column}: " + str(df_covid[column].max()) + "\n")

            print(f"Unique Values In {column}: {df_covid[column].unique()}")
            f.write(f"\nUnique Values In {column}: {df_covid[column].unique()} \n")

            print(f"Number of Nulls or NaNs in Column {column}: ", df_covid.iloc[index].isnull().sum())
            f.write(f"\nNumber of Nulls or NaNs in Column {column}: " + str(df_covid.iloc[index].isnull().sum()) + "\n")

            print(f"Breakdown of Values in Column {column} By Percentage. \n", df_covid[column].value_counts(normalize=True) * 100, "\n --------------- \n")
            f.write(f"\nBreakdown of Values in Column {column} By Percentage. \n" + str(df_covid[column].value_counts(normalize=True) * 100) + "\n --------------- \n")




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
    feature_analysis(df_covid)
    cleaned_df, df_rand, df_biased = remove_non_applicable_data(df_covid)
    clean_binary_columns(cleaned_df, df_rand, df_biased)
    print(df_rand)