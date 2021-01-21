import pickle

def load_pickle_file(path):
    try:
        with open(path, 'rb') as file:
            dataset = pickle.load(file)
        return dataset
    except FileNotFoundError as err:
        print("File not found! Please check your input filepath. Full error: ", err)


def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)