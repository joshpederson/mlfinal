import pickle

def unpickle(file) -> dict:
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data