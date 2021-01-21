
positive_path = "/Users/cburn/Data_Science_Playground/ML_In_Mexico/intubed_npy"
negative_path = "/Users/cburn/Data_Science_Playground/ML_In_Mexico/not_intubed_npy"
split_data_path = "/Users/cburn/Data_Science_Playground/ML_In_Mexico/split_data_dir"

train_split = 0.8
val_split = 0.1
test_split = 0.1

columns_of_interest = ['sex', 'patient_type', 'pneumonia', 'pregnancy', 'diabetes',
                       'copd', 'asthma', 'inmsupr', 'hypertension', 'other_disease',
                       'cardiovascular', 'obesity', 'renal_chronic', 'tobacco',
                       'contact_other_covid', 'covid_res', 'icu']