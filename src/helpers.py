import pickle
import os


def store_pickle(object, file_path):
    print("Storing pickle:", file_path)
    with open(file_path, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    if os.path.isfile(file_path):
        print("Loading pickle:", file_path)
        if os.path.getsize(file_path) > 0:
            with open(file_path, "rb") as handle:
                dic = pickle.load(handle)
            return dic
    else:
        return FileNotFoundError
