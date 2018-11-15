from easydict import EasyDict as edict

# the corresponding semantics to the index of
# obs.observation.feature_minimap and obs.observation.feature_screen

feature_mini_id = edict()
feature_mini_id.HEIGHT_MAP = 0
feature_mini_id.VISIBILITY = 1
feature_mini_id.CREEP = 2
feature_mini_id.CAMERA = 3
feature_mini_id.PLAYER_ID = 4
feature_mini_id.PLAYER_RELATIVE = 5
feature_mini_id.PLAYER_SELECTED = 6

feature_screen_id = edict()
feature_screen_id.HEIGHT_MAP = 0
feature_screen_id.VISIBILITY = 1
feature_screen_id.CREEP = 2
feature_screen_id.POWER = 3
feature_screen_id.PLAYER_ID = 4
feature_screen_id.PLAYER_RELATIVE = 5
feature_screen_id.UNIT_TYPE = 6
feature_screen_id.SELECTED = 7
feature_screen_id.HIT_POINTS = 8
feature_screen_id.ENERGY = 9
feature_screen_id.SHIELDS = 10
feature_screen_id.UNIT_DENSITY = 11
feature_screen_id.UNIT_DENSITY_AA = 12
