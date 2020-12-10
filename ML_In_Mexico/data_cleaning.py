import pandas as pd
import copy
import random
import numpy as np
from numpy import savetxt
from numpy import save

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
        df_covid.drop(df_covid[df_covid[col] > 3].index, inplace=True)
        df_covid.loc[df_covid[col] == 2, col] = 0

    # # This Dataframe randomly assigns 1 or 2 to rows which don't have values 1 or 2 in a column.
    # for col in binary_columns:
    #     df_random.loc[df_random[col] > 2, col] = random.choice([1, 2])
    #
    # # This Dataframe assumes that if an answer is not "Yes" to having a certain condition, then it is "No" for every
    # # column i.e. the pregnancy column.
    # for col in binary_columns:
    #     df_biased_to_no.loc[df_biased_to_no[col] > 1, col] = 2


    return df_covid, df_random, df_biased_to_no


def clean_binary_columns(fully_cleaned_df_covid, df_random, df_biased_to_no):
    df_list = [fully_cleaned_df_covid, df_random, df_biased_to_no]

    for dataframe in df_list:
        for col in binary_columns:
            dataframe.loc[dataframe[col] > 1, col] = 0


def drop_bad_columns(df_covid):
    # del df_covid['id']
    del df_covid['entry_date']
    del df_covid['date_symptoms']
    del df_covid['date_died']
    del df_covid['age']


# def drop_bad_label_rows(df_covid):
#     print(f"Dataframe Intubed Column (Length {len(df_covid)}) Before Drop: ")
#     print(f"Dataframe Intubed Column (Length {len(df_covid)}) After Drop: ")
#


def get_feature_vectors(columns, df_covid):
    # create new column
    # TAKE BAD ROWS OUT! TAKE OUT ANY ROW THAT DOESN"T HAVE INTUBED. Tka eout duplicate IDs.
    # SAVE TO DIfferent folders for intubed and not intubed.
    # USE DATASET ID AS filename
    feature_vector = {}

    print(df_covid)


    # Take each row in the dataframe...
    for index, row in enumerate(np.array(df_covid)):
        intubed = True if row[2] == 1 else False
        path = "not_intubed" if not intubed else "intubed"

        # Put in another sanity check here, and in all the feature vector functions.

        feature_vector = []

        feature_vector.extend(get_vector_sex(row[0]))
        feature_vector.extend(get_vector_patient_type(row[1]))
        feature_vector.extend(get_vector_pneumonia(row[3]))
        feature_vector.extend(get_vector_pregnancy(row[4]))
        feature_vector.extend(get_vector_diabetes(row[5]))
        feature_vector.extend(get_vector_copd(row[6]))
        feature_vector.extend(get_vector_asthma(row[7]))
        feature_vector.extend(get_vector_inmsupr(row[8]))
        feature_vector.extend(get_vector_hypertension(row[9]))
        feature_vector.extend(get_vector_other_disease(row[10]))
        feature_vector.extend(get_vector_cardiovascular(row[11]))
        feature_vector.extend(get_vector_obesity(row[12]))
        feature_vector.extend(get_vector_renal_chronic(row[13]))
        feature_vector.extend(get_vector_tobacco(row[14]))
        feature_vector.extend(get_vector_contact_other_covid(row[15]))
        feature_vector.extend(get_vector_covid_res(row[16]))
        feature_vector.extend(get_vector_icu(row[17]))

        # Capture the length of the feature vector.

        # save to csv file
        savetxt(f"/Users/cburn/Data_Science_Playground/ML_In_Mexico/{path}/row_{index}.csv", np.asarray(feature_vector), delimiter=',')
        save(f"/Users/cburn/Data_Science_Playground/ML_In_Mexico/{path}_npy/row_{index}.npy", np.asarray(feature_vector))
        print("VECTOR: ", feature_vector)

    #         for i, col in enumerate(columns):
    #             feature_vector[f"row_{index}"].extend(get_feature_vector(row[i], col, df_covid))
    #         print(f"Vector for row_{index}: ", feature_vector[f"row_{index}"])
    #
    # print("\n-----------------\n")
    # print(feature_vector)
    return df_covid


def get_vector_sex(item):
    # Male - 0, female - 1
    return [item]


def get_vector_patient_type(item):
    # Outpatient - 1, Inpatient - 0
    return [item]


def get_vector_pneumonia(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_pregnancy(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_diabetes(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_copd(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_asthma(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_inmsupr(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_hypertension(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_other_disease(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_cardiovascular(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_obesity(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_renal_chronic(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_tobacco(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_contact_other_covid(item):
    # Yes - 1, No - 0
    return [item]


def get_vector_covid_res(item):
    # Positive - 1, Negative - 0, Awaiting Results - 3
    if item == 0:
        return [0, 0]
    elif item == 1:
        return [1, 1]
    else:
        return [0, 1]


def get_vector_icu(item):
    # Yes - 1, No - 0
    return [item]

def get_feature_vector(column_value, col, df_covid):
    print(f"\n --- COLUMN {col} ---")

    unique_vals_in_column = sorted(list(df_covid[col].unique()))
    print("List of unique values: ", unique_vals_in_column)
    print(f"Index of {column_value}: ", unique_vals_in_column.index(column_value))

    index = unique_vals_in_column.index(column_value)
    binary_value = "{0:b}".format(index)
    print("Binary value of that: ", binary_value)

    return [int(char) for char in binary_value]


def clean_data():
    df_covid = import_data()
    feature_analysis(df_covid)

    drop_bad_columns(df_covid)
    # drop_bad_label_rows(df_covid)

    print(df_covid.columns)
    columns = ['sex', 'patient_type', 'pneumonia', 'pregnancy', 'diabetes',
               'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease',
               'cardiovascular', 'obesity', 'renal_chronic', 'tobacco',
               'contact_other_covid', 'covid_res', 'icu']

    cleaned_df, df_rand, df_biased = remove_non_applicable_data(df_covid)
    features = get_feature_vectors(columns, cleaned_df)


if __name__ == "__main__":
    df_covid = import_data()
    feature_analysis(df_covid)

    drop_bad_columns(df_covid)
    # drop_bad_label_rows(df_covid)

    print(df_covid.columns)
    columns = ['sex', 'patient_type', 'pneumonia', 'pregnancy', 'diabetes',
       'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease',
       'cardiovascular', 'obesity', 'renal_chronic', 'tobacco',
       'contact_other_covid', 'covid_res', 'icu']

    cleaned_df, df_rand, df_biased = remove_non_applicable_data(df_covid)
    features = get_feature_vectors(columns, cleaned_df)



    # clean_binary_columns(cleaned_df, df_rand, df_biased)
    # print(df_rand)