import pandas as pd
import copy
import random
import numpy as np
from numpy import savetxt
from numpy import save
import datetime
from sklearn import preprocessing

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


def perform_feature_analysis(df_covid):
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

    columns_to_target = ["sex", "patient_type", "pneumonia", "diabetes", "asthma", "hypertension", "obesity",
                         "tobacco"]
    # This Dataframe drops any rows that don't have values 1 or 2 - and so, will not be very large.
    for col in columns_to_target:
        df_covid.drop(df_covid[df_covid[col] > 3].index, inplace=True)
        df_covid.loc[df_covid[col] == 2, col] = 0


    # Scale the date columns.
    x = df_covid.values  # returns a numpy array
    scaler = preprocessing.MinMaxScaler()
    df_covid[["days_until_died_from_being_hospitalized", "days_until_died_from_being_hospitalized"]] = \
        scaler.fit_transform(df_covid[["days_until_died_from_being_hospitalized", "days_until_died_from_being_hospitalized"]])

    return df_covid



    # # This Dataframe randomly assigns 1 or 2 to rows which don't have values 1 or 2 in a column.
    # for col in binary_columns:
    #     df_random.loc[df_random[col] > 2, col] = random.choice([1, 2])
    #
    # # This Dataframe assumes that if an answer is not "Yes" to having a certain condition, then it is "No" for every
    # # column i.e. the pregnancy column.
    # for col in binary_columns:
    #     df_biased_to_no.loc[df_biased_to_no[col] > 1, col] = 2


    return df_covid


def clean_binary_columns(fully_cleaned_df_covid, df_random, df_biased_to_no):
    df_list = [fully_cleaned_df_covid, df_random, df_biased_to_no]

    for dataframe in df_list:
        for col in binary_columns:
            dataframe.loc[dataframe[col] > 1, col] = 0


def drop_bad_columns(df_covid, columns_to_target=None):
    if not columns_to_target:
        # del df_covid['id']
        del df_covid['entry_date']
        del df_covid['date_symptoms']
        del df_covid['date_died']
        del df_covid['age']
    else:
        df_covid = df_covid[df_covid.columns.intersection(columns_to_target)]
    return df_covid

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

    print(df_covid.columns)

    # RENAME FILE NAMES TO IDs OF FILES RATHER THAN row_x!


    i = {'intubed': 0, 'not_intubed': 0}
    # Take each row in the dataframe...
    for index, row in enumerate(np.array(df_covid)):
        row = row[1:]
        intubed = True if row[2] == 1 else False
        if intubed:
            i['intubed'] += 1
        else:
            i['not_intubed'] += 1
        print(row)
        print(row[0])
        print(row[2])

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
        feature_vector.extend([row[18]])
        feature_vector.extend([row[19]])

        # Capture the length of the feature vector.

        # save to csv file
        # savetxt(f"/Users/cburn/Data_Science_Playground/ML_In_Mexico/{path}/row_{index}.csv", np.asarray(feature_vector), delimiter=',')
        save(f"/Users/cburn/Data_Science_Playground/ML_In_Mexico/{path}_npy/row_{index}.npy", np.asarray(feature_vector))
        print("VECTOR: ", feature_vector)

    #         for i, col in enumerate(columns):
    #             feature_vector[f"row_{index}"].extend(get_feature_vector(row[i], col, df_covid))
    #         print(f"Vector for row_{index}: ", feature_vector[f"row_{index}"])
    #
    # print("\n-----------------\n")
    # print(feature_vector)
    print(i)
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
    perform_feature_analysis(df_covid)

    drop_bad_columns(df_covid)
    # drop_bad_label_rows(df_covid)

    print(df_covid.columns)
    columns = ['sex', 'patient_type', 'pneumonia', 'pregnancy', 'diabetes',
               'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease',
               'cardiovascular', 'obesity', 'renal_chronic', 'tobacco',
               'contact_other_covid', 'covid_res', 'icu', 'days_until_hospitalized', 'days_until_died_from_being_hospitalized']

    cleaned_df, df_rand, df_biased = remove_non_applicable_data(df_covid)
    features = get_feature_vectors(columns, cleaned_df)

def convert(x):
    try:
        return datetime.datetime.strptime(str(x['date_died']), '%d-%m-%Y').date()
    except ValueError as err:
        return None


def create_new_columns(df_covid):
    import datetime
    from datetime import date
    days_until_hospitalized = 1
    print(df_covid['date_symptoms'][0])
    df_covid['date_symptoms'] = df_covid['date_symptoms'].apply(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').date())

    df_covid['entry_date'] = df_covid['entry_date'].apply(
        lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').date())

    temp_df = df_covid[['entry_date', 'date_died']]
    df_covid['date_died'] = temp_df.apply(lambda x: convert(x), axis=1)

    df_covid['days_until_hospitalized'] = (df_covid['entry_date'] - df_covid['date_symptoms']).dt.days
    df_covid['days_until_died_from_being_hospitalized'] = df_covid['date_died'].dt.days - df_covid['entry_date'].dt.days

    print(df_covid['date_symptoms'][0], df_covid['entry_date'][0], df_covid['days_until_hospitalized'][0], df_covid['days_until_died_from_being_hospitalized'][0])
    print(df_covid['days_until_hospitalized'])
    print(df_covid['days_until_died_from_being_hospitalized'])

    print(df_covid)
    print(df_covid.columns)



if __name__ == "__main__":
    # columns = ['sex', 'patient_type', 'pneumonia', 'pregnancy', 'diabetes',
    #    'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease',
    #    'cardiovascular', 'obesity', 'renal_chronic', 'tobacco',
    #    'contact_other_covid', 'covid_res', 'icu']

    columns_to_target = ["id", "sex", "patient_type", "pneumonia", "diabetes", "asthma", "hypertension", "obesity",
                         "tobacco", "days_until_died_from_being_hospitalized", "days_until_died_from_being_hospitalized"]
    df_covid = import_data()
    # perform_feature_analysis(df_covid)
    create_new_columns(df_covid)

    df_covid = drop_bad_columns(df_covid, columns_to_target)
    # drop_bad_label_rows(df_covid)

    print(df_covid.columns)


    cleaned_df = remove_non_applicable_data(df_covid)
    pd.set_option('display.max_columns', None)
    print(cleaned_df.head(10))

    # TAKE A LOOK AT % OF ZEROES AND ONES
    
    # features = get_feature_vectors(columns, cleaned_df)
