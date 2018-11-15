from lib.enums import feature_mini_id, feature_screen_id

def local_index2local_ratio(xy, obs):
    observation = obs.observation
    feature_screen = observation.feature_screen
    screen_height, screen_width = feature_screen[feature_screen_id.HEIGHT_MAP].shape

    x, y = xy
    return [float(x) / screen_height, float(y) / screen_width]

def local_ratio2local_index(xy_ratio, obs):
    observation = obs.observation
    feature_screen = observation.feature_screen
    screen_height, screen_width = feature_screen[feature_screen_id.HEIGHT_MAP].shape

    x_ratio, y_ratio = xy_ratio
    return [int(x_ratio * screen_height), int(y_ratio * screen_width)]

def global_index2global_ratio(xy, obs):
    observation = obs.observation
    feature_minimap = observation.feature_minimap
    selected_map = feature_minimap[feature_mini_id.CAMERA]
    map_height, map_width = selected_map.shape

    x, y = xy
    return [float(x) / map_height, float(y) / map_width]

def global_ratio2global_index(xy_ratio, obs):
    observation = obs.observation
    feature_minimap = observation.feature_minimap
    selected_map = feature_minimap[feature_mini_id.CAMERA]
    map_height, map_width = selected_map.shape

    x_ratio, y_ratio = xy_ratio
    return [int(x_ratio * map_height), int(y_ratio * map_width)]


def local_ratio2global_ratio(xy_ratio, obs):
    observation = obs.observation
    feature_minimap = observation.feature_minimap
    selected_map = feature_minimap[feature_mini_id.CAMERA]
    x_ratio, y_ratio = xy_ratio

    x_min = 10000
    y_min = 10000
    x_max = 0
    y_max = 0

    for i in range(selected_map.shape[0]):
        for j in range(selected_map.shape[1]):
            if selected_map[i][j]:
                x_min = min(i, x_min)
                y_min = min(j, y_min)
                x_max = max(i, x_max)
                y_max = max(j, y_max)

    camera_height = x_max - x_min + 1
    camera_width = y_max - y_min + 1

    camera_top = float(x_min) / selected_map.shape[0]
    camera_left = float(y_min) / selected_map.shape[1]

    local_offset_x = x_ratio * float(camera_height) / selected_map.shape[0]
    local_offset_y = y_ratio * float(camera_width) / selected_map.shape[1]

    return [camera_top + local_offset_x, camera_left + local_offset_y]

def global_ratio2local_ratio(xy, obs):
    observation = obs.observation
    feature_minimap = observation.feature_minimap
    selected_map = feature_minimap[feature_mini_id.CAMERA]

    global_x, global_y = xy

    x_min = 10000
    y_min = 10000
    x_max = 0
    y_max = 0

    for i in range(selected_map.shape[0]):
        for j in range(selected_map.shape[1]):
            if selected_map[i][j]:
                x_min = min(i, x_min)
                y_min = min(j, y_min)
                x_max = max(i, x_max)
                y_max = max(j, y_max)

    camera_height = x_max - x_min + 1
    camera_width = y_max - y_min + 1

    camera_top = float(x_min) / selected_map.shape[0]
    camera_left = float(y_min) / selected_map.shape[1]

    local_offset_x = (global_x - camera_top) * selected_map.shape[0] / camera_height
    local_offset_y = (global_y - camera_left) * selected_map.shape[1] / camera_width

    return [local_offset_x, local_offset_y]
