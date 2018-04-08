import pickle
import os


def store_pickle(obj, file_path):
    print("Storing pickle:", file_path)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    if os.path.isfile(file_path):
        print("Loading pickle:", file_path)
        if os.path.getsize(file_path) > 0:
            with open(file_path, "rb") as handle:
                dic = pickle.load(handle)
            return dic
    else:
        return FileNotFoundError


# creates absolute path
def abspath(path, *paths):
    fpath = os.path.join(os.getcwd(), os.pardir, path)

    for p in paths:
        fpath = os.path.join(fpath, p)
    return fpath