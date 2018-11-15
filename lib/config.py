"""
save some information of configuration by easydict
"""
from easydict import EasyDict as edict

config = edict()
# the information about the shape of the building by lib.building
config.BUILDING_CACHE_PATH = r"./cache/building_shape"
