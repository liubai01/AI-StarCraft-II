import pickle

def load(path):
    with open(path, "rb") as f:
        ret = pickle.load(f)
    return ret

print(load(r"./cache/building_shape/18.pkl"))
