from lib.enums import feature_mini_id, feature_screen_id
from lib.building import load_building_template_by_id

import numpy as np


__all__ = ["get_building_center", "find_units_by_id"]

def find_units_by_id(obs, uid):
    """
    output a list  contain the coordinations of the pixels occupied by target
    unit in local camera
    :param obs: the observation provided in baseAgent.step as input
    :param uid: id of that unit(defined in pysc2.lib.units)
    :return: the valid unit coordinations in list
    xys = [<x1, y1>, <x2, y2>, ...]
    """
    ret = []

    observation = obs.observation
    feature_screen = observation.feature_screen

    feature_unit = feature_screen[feature_screen_id.UNIT_TYPE]
    for i in range(feature_unit.shape[0]):
        for j in range(feature_unit.shape[1]):
            if feature_unit[i][j] == uid:
                ret.append([i, j])

    return ret

def xys2grid(xys):
    x_min = 10000
    y_min = 10000
    x_max = 0
    y_max = 0

    for x, y in xys:
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)

    ret_grid = np.zeros(
        (x_max - x_min + 1, y_max - y_min + 1),
    )
    for x, y in xys:
        ret_grid[x - x_min][y - y_min] = 1

    return ret_grid, x_min, x_max, y_min, y_max

def grid2xys(grid, x_min, y_min):
    ret = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                ret.append([i + x_min, j + y_min])
    return ret

VALID_BUILDING_THRESHOLD = 0.9

def get_building_center(obs, uid):
    """
    get the center of the building
    :param obs: the observation provided in baseAgent.step as input
    :param uid: id of that building(defined in pysc2.lib.units)
    :return: the coordination of the target building which is almost completely in the camera
    e.g: [(x1, y1), (x2, y2), ...]
    """
    ret = []
    xys = find_units_by_id(obs, uid)
    template = load_building_template_by_id(uid)

    while len(xys) >= len(template) * VALID_BUILDING_THRESHOLD:
        grid, x_min, x_max, y_min, y_max = xys2grid(xys)
        c = 0.
        c_x = 0.
        c_y = 0.
        for x, y in template:
            try:
                if grid[x][y] == 1:
                    grid[x][y] = 0
                    c += 1
                    c_x += x_min + x
                    c_y += y_min + y
            except IndexError:
                pass
        if c / len(template) >= VALID_BUILDING_THRESHOLD:
            ret.append([c_x / c, c_y / c])
        if c == 0:
            break
        xys = grid2xys(grid, x_min, y_min)

    return ret
