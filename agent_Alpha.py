from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions, units
from lib.parsers.obs import get_building_center
from lib.parsers.coordinate_transfer import *
from lib.building import save_single_building_shape

import time

class alphaAgent(base_agent.BaseAgent):

  def step(self, obs):
    super(alphaAgent, self).step(obs)
    observation = obs.observation

    # detect the CommandCenter and grasp its center(can deal with multiple center)
    time_start = time.time()
    c = get_building_center(obs, units.Terran.CommandCenter)
    print(c)
    time_end = time.time()
    print(time_end - time_start)

    # if len(c) > 0:
    #     sample_c_ratio = local_index2local_ratio(c[0], obs)
    #     global_xy = local_ratio2global_ratio(sample_c_ratio, obs)
    #     recover_xy = global_ratio2local_ratio(global_xy, obs)
    #     print(sample_c_ratio, global_xy, recover_xy)
    #     print(local_ratio2local_index(recover_xy, obs))

    time.sleep(1)
    return actions.FunctionCall(0, []) #idle
