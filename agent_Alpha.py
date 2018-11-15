from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions, units
from lib.enums import feature_mini_id, feature_screen_id
from lib.parsers.obs import obs2building_center
from lib.parsers.coordinate_transfer import *
from lib.building import save_single_building_shape

import time

id_max = 0

class alphaAgent(base_agent.BaseAgent):

  def step(self, obs):
    super(alphaAgent, self).step(obs)
    observation = obs.observation

    time_start = time.time()
    c = obs2building_center(obs, units.Terran.CommandCenter)
    time_end = time.time()
    print(time_end - time_start)

    if len(c) > 0:
        sample_c_ratio = local_index2local_ratio(c[0], obs)
        global_xy = local_ratio2global_ratio(sample_c_ratio, obs)
        recover_xy = global_ratio2local_ratio(global_xy, obs)
        print(sample_c_ratio, global_xy, recover_xy)
        print(local_ratio2local_index(recover_xy, obs))

    time.sleep(1)
    print(c)
    # time.sleep(1)
    return actions.FunctionCall(0, [])
