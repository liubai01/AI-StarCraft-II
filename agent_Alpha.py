from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions, units
from scNet import scNet, scNetOutput2candidateAction
from lib.parsers.obs import find_units_by_id
from lib.pysc2_info_saver import recorder
from torch import optim

import numpy as np
from pysc2.lib import features
import time
import torch
import os
import random

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

class idleAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(idleAgent, self).step(obs)
        time.sleep(0.02)
        return FUNCTIONS.no_op()

class dummyAgent(base_agent.BaseAgent):

    def __init__(self):
        super(dummyAgent, self).__init__()
        self.previous_reward = 0
        self.custom_reward = 0
        self.my_accumlated_reward = 0
        self.my_step = 0
        self.has_started_game = False
        self.now_action_id = None
        self.recorder = recorder()

    def step(self, obs):
        super(dummyAgent, self).step(obs)
        self.previous_reward = self.custom_reward
        self.my_step += 1
        if obs.reward == 0:
            # avoid confliction with built-in reward
            self.custom_reward = 0
        else:
            self.custom_reward = obs.reward
        self.custom_reward -= 3 # time elpase
        self.custom_reward -= len(obs.observation.action_result) * 10 # error action punishment
        if self.now_action_id == 0:
            self.custom_reward + 1 # avoid overhead
        self.custom_reward  = float(self.custom_reward) / 10 # reduce by constant factor (for numerically stable)
        self.my_accumlated_reward += self.custom_reward

        # generate the feature for fully-connected input
        # section(do not contain the spatial information
        self.feature_fc = []
        # add previous time stamp reward into that
        self.feature_fc.append(self.previous_reward)
        # add selected information into feature fc (max 20)
        selected_info = obs.observation.multi_select
        selected_info_clip = np.zeros((20, 3))
        if selected_info.shape[0] != 0:
            cache_num = min(selected_info.shape[0] ,20)
            selected_info_clip[:cache_num, :] = np.array(selected_info)[:cache_num, :3]
        self.feature_fc += list(selected_info_clip.reshape(-1))
        # add player general infos
        self.feature_fc.append(obs.observation.player[1]) # minerals
        self.feature_fc.append(obs.observation.player[8]) # army count
        self.feature_fc = np.array([self.feature_fc]) / float(100)

        # generate the features for 2d input (minimap)
        self.feature_minimap = [
            obs.observation.feature_minimap[3], # camera
            obs.observation.feature_minimap[5], # player_relative
            obs.observation.feature_minimap[6]  # selected
        ]
        self.feature_minimap = np.array([self.feature_minimap]) / float(100)
        # generate the features for 2d input (screen)
        self.feature_screen = [
            obs.observation.feature_screen[1],
            obs.observation.feature_screen[5],
            obs.observation.feature_screen[6],
            obs.observation.feature_screen[7],
            obs.observation.feature_screen[8],
            obs.observation.feature_screen[9],
            obs.observation.feature_screen[14],
            obs.observation.feature_screen[15],
        ]
        self.feature_screen = np.array([self.feature_screen]) / float(100)

        # transform into cuda torch
        self.feature_fc = torch.from_numpy(self.feature_fc)
        self.feature_minimap = torch.from_numpy(self.feature_minimap)
        self.feature_screen = torch.from_numpy(self.feature_screen)

        self.has_started_game = True
        return actions.FunctionCall(0, [])

    def reset(self):
        super(dummyAgent, self).reset()
        self.recorder.dump(self.episodes, self.my_accumlated_reward)
        self.my_step = 0
        self.my_accumlated_reward = 0

class simpleContestAgent(dummyAgent):
      def __init__(self):
          super(simpleContestAgent, self).__init__()
          if os.path.exists("model.pkl"):
              self.model = torch.load("model.pkl")
              self.model.cuda()
          else:
              self.model = scNet()
              self.model.cuda()
          self.D = []
          self.N = 1200
          self.previous_state = None
          self.now_state = None
          self.batch_num = 100
          self.gamma = 0.95
          self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)

      def step(self, obs):
          super(simpleContestAgent, self).step(obs)
          self.feature_fc = self.feature_fc.float().cuda()
          self.feature_screen = self.feature_screen.float().cuda()
          self.feature_minimap = self.feature_minimap.float().cuda()
          self.now_valid_actions = obs.observation.available_actions

          model_output = self.model(
              self.feature_fc,
              self.feature_screen,
              self.feature_minimap
          )
          # hidden_state = self.model.get_hidden()
          # self.model.dump_hidden(hidden_state)
          self.now_action_id, ret_action, _ = self.scNet2action(obs, model_output)
          if self.turn:
              self.update_D()
              if self.my_step % 200 == 0:
                  self.train()
          # if self.now_action_id in [331, 332]:
          #   print(self.now_action_id)
          return ret_action

      def scNet2action(self, obs, model_output, eplison=0.9):
          model_output = model_output
          flip_coin = random.random()

          actions = scNetOutput2candidateAction(model_output, flip_coin > eplison)

          valid_actions = obs.observation.available_actions
          candidate_actions_id = [a["action_id"] for a in actions]

          ret_action = None
          ret_QValue = None

          if flip_coin < eplison:
              for a in actions:
                  if a["action_id"] in valid_actions:
                      ret_action_id = a["action_id"]
                      ret_action = a["action"]
                      ret_QValue = a["Q_value"]
                      break
          else:
              ok_id = list(set(valid_actions).intersection(set(candidate_actions_id)))
              id = random.choice(ok_id)
              for a in actions:
                  if a["action_id"] == id:
                      ret_action_id = a["action_id"]
                      ret_action = a["action"]
                      ret_QValue = a["Q_value"]
                      break
          assert(not (ret_action is None))
          return ret_action_id, ret_action, ret_QValue

      def get_state(self):
          save_dict = {}
          save_dict["feature_fc"] = self.feature_fc.cpu()
          save_dict["feature_screen"] = self.feature_screen.cpu()
          save_dict["feature_minimap"] = self.feature_minimap.cpu()
          save_dict["valid_actions"] = self.now_valid_actions
          save_dict["hidden_states"] = self.model.get_hidden()
          save_dict["reward"] = self.custom_reward
          save_dict["action_id"] = self.now_action_id

          return save_dict

      def update_D(self):
          D_sample = {}
          self.now_state = self.get_state()
          if not self.previous_state is None:
              D_sample["now"] = self.previous_state
              D_sample["next"] = self.now_state
              D_sample["terminal"] = False
              # evict one if full
              if len(self.D) == self.N:
                  i = random.randint(0, len(self.D) - 1)
                  s_evicted = self.D.pop(i)
                  for it in s_evicted.items():
                      del(it)
                  del(i)
              self.D.append(D_sample)
          self.previous_state = self.now_state

      def train(self):
          if len(self.D) == 0:
              return
          batch_num = min(self.batch_num, len(self.D))
          samples = random.sample(self.D, batch_num)
          # cache tmp_hidden_states
          tmp_hidden_states = self.model.get_hidden()
          losses = 0
          for s in samples:
              reward = s["now"]["reward"]
              candidate_Q_values = []
              if s["terminal"]:
                  print("train the terminal state already!")
                  y = torch.from_numpy(np.array([reward]))[0]
              else:
                  # get max_a Q(phi(j+1), a;theta)
                  self.model.dump_hidden(s["next"]["hidden_states"])
                  model_output = self.model(
                      s["next"]["feature_fc"].cuda(),
                      s["next"]["feature_screen"].cuda(),
                      s["next"]["feature_minimap"].cuda()
                  )
                  s["next"]["feature_fc"].cpu()
                  s["next"]["feature_screen"].cpu()
                  s["next"]["feature_minimap"].cpu()
                  actions = scNetOutput2candidateAction(model_output)
                  valid_actions = s["next"]["valid_actions"]
                  for a in actions:
                      if a["action_id"] in valid_actions:
                          max_Q_value = a["Q_value"].detach()
                          break
                  y = reward + self.gamma * max_Q_value
              self.model.dump_hidden(s["now"]["hidden_states"])
              model_output = self.model(
                  s["now"]["feature_fc"].cuda(),
                  s["now"]["feature_screen"].cuda(),
                  s["now"]["feature_minimap"].cuda()
              )
              s["now"]["feature_fc"].cpu()
              s["now"]["feature_screen"].cpu()
              s["now"]["feature_minimap"].cpu()
              actions = scNetOutput2candidateAction(model_output)
              for a in actions:
                  if a["action_id"] == s["now"]["action_id"]:
                      now_Q_value = a["Q_value"]
                      break
              loss = torch.abs(y.float().cuda() - now_Q_value)
              losses += loss
          losses /= batch_num
          print("eposide:{}, step:{}, loss: {}".format(self.episodes,
                                                       self.my_step,
                                                       float(losses.cpu().detach()))
                )
          self.optimizer.zero_grad()
          losses.backward()
          for _ in range(20):
            self.optimizer.step()

          self.model.dump_hidden(tmp_hidden_states)

      def reset(self):
          super(simpleContestAgent, self).reset()
          if self.turn:
              torch.save(self.model, "model.pkl")
              print("model saved successfully")
          else:
              if os.path.exists("model.pkl"):
                self.model = torch.load("model.pkl")
          self.previous_state = None


      def resetAndGetEnv(self, env, player_id):
          super(simpleContestAgent, self).resetAndGetEnv(env, player_id)
          print("player_{} | trainable: {}".format(int(player_id), self.turn))
          if self.has_started_game:
              D_sample = {}
              self.now_state = self.get_state()
              D_sample["now"] = self.now_state
              D_sample["now"]["reward"] += env.outcome[player_id] * 800. / 10
              D_sample["next"] = None
              D_sample["terminal"] = True
              # evict one if full
              if len(self.D) == self.N:
                  i = random.randint(0, len(self.D) - 1)
                  s_evicted = self.D.pop(i)
                  for it in s_evicted.items():
                      del(it)
                  del(i)
              self.D.append(D_sample)
              print("has added terminal state, {} {}".format(D_sample["terminal"], D_sample["now"]["reward"]))
              self.train()



# class fixedAgent(dummyAgent):
#       def __init__(self):
#           super(fixedAgent, self).__init__()
#           if os.path.exists("model.pkl"):
#               self.model = torch.load("model.pkl")
#               self.model.cuda()
#           else:
#               self.model = scNet()
#               self.model.cuda()
#       def step(self, obs):
#           super(fixedAgent, self).step(obs)
#           self.feature_fc = self.feature_fc.float().cuda()
#           self.feature_screen = self.feature_screen.float().cuda()
#           self.feature_minimap = self.feature_minimap.float().cuda()
#           self.now_valid_actions = obs.observation.available_actions
#
#           model_output = self.model(
#               self.feature_fc,
#               self.feature_screen,
#               self.feature_minimap
#           )
#           _, ret_action, _ = self.scNet2action(obs, model_output)
#           return ret_action
#
#       def scNet2action(self, obs, model_output, eplison=0.7):
#           actions = scNetOutput2candidateAction(model_output)
#
#           valid_actions = obs.observation.available_actions
#           candidate_actions_id = [a["action_id"] for a in actions]
#
#           ret_action = None
#           ret_QValue = None
#           flip_coin = random.random()
#
#           if flip_coin < eplison:
#               for a in actions:
#                   if a["action_id"] in valid_actions:
#                       ret_action_id = a["action_id"]
#                       ret_action = a["action"]
#                       ret_QValue = a["Q_value"]
#                       break
#           else:
#               ok_id = list(set(valid_actions).intersection(set(candidate_actions_id)))
#               id = random.choice(ok_id)
#               for a in actions:
#                   if a["action_id"] == id:
#                       ret_action_id = a["action_id"]
#                       ret_action = a["action"]
#                       ret_QValue = a["Q_value"]
#                       break
#           assert(not (ret_action is None))
#           return ret_action_id, ret_action, ret_QValue
#
#       def reset(self):
#           super(fixedAgent, self).reset()
#           self.model = torch.load("model.pkl")
#           print("model saved successfully")
# a f2a robot with continously training
class botAgent(base_agent.BaseAgent):
    def _xy_locs(self, mask):
      """Mask should be a set of bools from comparison with a feature layer."""
      y, x = mask.nonzero()
      return list(zip(x, y))
    def step(self, obs):
        xys = find_units_by_id(obs, units.Terran.Barracks)
        if not xys is None and len(xys) > 0 and obs.observation.player[1] >= 50:
            x, y = random.choice(xys)
            if not 477 in obs.observation.available_actions:
                return actions.FunctionCall(2, [[0], [y, x]])
            else:
                return actions.FunctionCall(477, [[True]])
        if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
          player_relative = obs.observation.feature_screen.player_relative
          enemy = self._xy_locs(player_relative == _PLAYER_ENEMY)
          if not enemy:
            return FUNCTIONS.no_op()
          target = enemy[np.argmax(np.array(enemy)[:, 1])]
          return FUNCTIONS.Attack_screen("now", target)

        if FUNCTIONS.select_army.id in obs.observation.available_actions:
          return FUNCTIONS.select_army("select")

        return FUNCTIONS.no_op()

# a robot for testing observations
class betaAgent(base_agent.BaseAgent):

      def __init__(self):
            super(betaAgent, self).__init__()
            self.total_step = 0

      def step(self, obs):
            super(betaAgent, self).step(obs)
            if obs.reward == 0:
                reward = 0
            else:
                reward = obs.reward
                if reward % 50 != 0:
                    reward -= 10
            # if obs.reward != 0:
            #     print("obs:{}".format(obs.reward))
            if reward != 0:
                print(reward)
            # if obs.reward != 0 and abs(obs.reward) >= 50:
            #     print(obs.reward)
            # print(obs.observation.feature_screen.shape)
            # # print(obs.observation.multi_select)


            assert obs.reward >= -float("inf")

            time.sleep(0.05)

            # if 2 in obs.observation.available_actions:
            #     return actions.FunctionCall(2, [[2],[40, 40]])
            return actions.FunctionCall(0, [])
            # return actions.FunctionCall(19, [[True], [40, 40]])
      def reset(self):
            super(betaAgent, self).reset()

