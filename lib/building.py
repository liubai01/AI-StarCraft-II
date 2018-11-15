import pickle
import copy
import os
from lib.config import config

def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def save_single_building_shape(xys, uid):
    xys = copy.deepcopy(xys)
    x_min = 10000
    y_min = 10000
    for x, y in xys:
        x_min = min(x, x_min)
        y_min = min(y, y_min)

    for i in range(len(xys)):
        xys[i][0] -= x_min
        xys[i][1] -= y_min

    filename = "{}.pkl".format(uid)
    save(xys, os.path.join(config.BUILDING_CACHE_PATH, filename))

def load_building_template_by_id(uid):
    filename = "{}.pkl".format(uid)
    try:
        return load(os.path.join(config.BUILDING_CACHE_PATH, filename))
    except FileNotFoundError:
        raise Exception("Template for building[{}] not founded".format(uid))
