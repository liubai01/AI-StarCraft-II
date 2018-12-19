from torch import nn
from lib.layers.Replicate_unit import Replicate_unit1d
from lib.layers.FC_LSTM_unit import FC_lstm_stacked
from lib.layers.Conv_LSTM_unit import CONV_lstm_unit
import random
import torch.nn.functional as F
import torch
from pysc2.lib import actions
import numpy as np

class scNet(nn.Module):
    def __init__(self):
        super(scNet,self).__init__()
        # 1. memoryless units section
        self.conv_screen_1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv_screen_2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1)

        self.conv_minimap_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv_minimap_2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1)

        self.padding_minimap = torch.nn.ZeroPad2d((2, 3, 2, 3, 0, 0))

        self.fc1 = nn.Linear(63, 64)
        self.fc2 = nn.Linear(64, 128)

        # 2. memory decision section
        self.fc2conv_replicate = Replicate_unit1d(21, 21)
        self.conv_combined = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

        self.FC_mem = FC_lstm_stacked([569, 24, 128], 2)
        # self.CONV_mem = CONV_lstm_unit(128, 64, 3, 5)
        # self.FC_mem = nn.Linear(569, 128)
        self.CONV_mem = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        # 3. memoryless output decision section
        self.fc1_decision = nn.Linear(569, 32)
        self.fc2_decision = nn.Linear(32, 28)

        self.fc2conv_mini = Replicate_unit1d(16, 16)
        self.fc2conv_screen = Replicate_unit1d(21, 21)

        self.dec_conv_mini = nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, padding=1)
        self.dec_conv_screen = nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, padding=1)

        self.dec_deconv_mini1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec_deconv_mini2 = nn.ConvTranspose2d(32, 3, 2, 2)

        self.dec_deconv_screen1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec_deconv_screen2 = nn.ConvTranspose2d(32, 6, 2, 2)

    def forward(self, feature_fc, feature_screen, feature_minimap):
        batch_num = feature_screen.size()[0]
        # 1. memoryless units section

        # 1.1 deal with 2d input
        feature_screen_1 = F.max_pool2d(F.relu(self.conv_screen_1(feature_screen)), 2)
        feature_screen_2 = F.max_pool2d(F.relu(self.conv_screen_2(feature_screen_1)), 2)

        feature_minimap_1 = F.max_pool2d(F.relu(self.conv_minimap_1(feature_minimap)), 2)
        feature_minimap_2 = F.max_pool2d(F.relu(self.conv_minimap_2(feature_minimap_1)), 2)

        assert(list(feature_screen_2.size()[1:]) == [32, 21, 21])
        assert(list(feature_minimap_2.size()[1:]) == [32, 16, 16])

        # padding with zero to have the same size as feature_screen
        feature_minimap_padded = self.padding_minimap(feature_minimap_2)
        assert(list(feature_minimap_padded.size()[1:]) == [32, 21, 21])

        feature_combined = torch.cat((feature_screen_2, feature_minimap_padded), dim=1)
        assert(list(feature_combined.size()[1:]) == [64, 21, 21])

        # 2. deal with 1-d input(general information/selected information)
        feature_fc_1 = F.relu(self.fc1(feature_fc))
        feature_fc_2 = F.relu(self.fc2(feature_fc_1))
        assert(list(feature_fc_2.size()[1:]) == [128])

        # 2. memory decision section
        # 2.1 cross the infomation of fc and conv
        # 2.1.1 add fc information to conv
        feature_fc2conv = self.fc2conv_replicate(feature_fc_2[:, 64:])
        feature_conv_mem = torch.cat([feature_combined, feature_fc2conv], dim=1)
        assert(list(feature_conv_mem.size()[1:]) == [128, 21, 21])
        # 2.1.2 add conv information to fc
        feature_combined_compress = F.relu(self.conv_combined(feature_combined))
        feature_combined_flatten = feature_combined_compress.view(batch_num, -1)
        feature_fc_mem = torch.cat([feature_fc_2, feature_combined_flatten], dim=1)
        assert(list(feature_fc_mem.size()[1:]) == [569])
        # 2.2 fc lstm
        feature_fc_mem_out = self.FC_mem(feature_fc_mem)
        assert(list(feature_fc_mem_out.size()[1:]) == [128])
        # 2.3 conv lstm
        feature_conv_mem_out = self.CONV_mem(feature_conv_mem)
        assert(list(feature_conv_mem_out.size()[1:]) == [64, 21, 21])

        # 3. memoryless output decision section
        # 3.1 FC section
        # 3.1.1 combine with the memory conv output
        feature_fc_decision_in = torch.cat([feature_fc_mem_out, feature_conv_mem_out[:, 63].view(batch_num, 21 * 21)], dim=1)
        assert(list(feature_fc_decision_in.size()[1:]) == [569])

        feature_fc_decision_1 = F.relu(self.fc1_decision(feature_fc_decision_in))
        feature_fc_decision_out = F.tanh(self.fc2_decision(feature_fc_decision_1)) * 10

        assert(list(feature_fc_decision_out.size()[1:]) == [28])
        # 3.2 conv section
        # 3.2.1 split & crop & stack
        # (1) split
        feature_conv_dec_screen = feature_conv_mem_out[:, :32]
        feature_conv_dec_minimap = feature_conv_mem_out[:, 32:]
        # (2) crop
        feature_conv_dec_minimap = feature_conv_dec_minimap[:, :, 2: 18, 2: 18]
        # (3) stack
        # stack with fc memory output
        feature_conv_dec_fcstack_screen = self.fc2conv_screen(feature_fc_mem_out[:, -32:])
        feature_conv_dec_screen = torch.cat([feature_conv_dec_screen, feature_conv_dec_fcstack_screen], dim=1)
        feature_conv_dec_fcstack_minimap = self.fc2conv_mini(feature_fc_mem_out[:, -32:])
        feature_conv_dec_minimap = torch.cat([feature_conv_dec_minimap, feature_conv_dec_fcstack_minimap], dim=1)
        # stack with memoryless units section
        feature_conv_dec_fcstack_screen = feature_screen_2[:, -16:]
        feature_conv_dec_screen = torch.cat([feature_conv_dec_screen, feature_conv_dec_fcstack_screen], dim=1)
        feature_conv_dec_fcstack_minimap = feature_minimap_2[:, -16:]
        feature_conv_dec_minimap = torch.cat([feature_conv_dec_minimap, feature_conv_dec_fcstack_minimap], dim=1)

        assert(list(feature_conv_dec_minimap.size()[1:]) == [80, 16, 16])
        assert(list(feature_conv_dec_screen.size()[1:]) == [80, 21, 21])

        # 3.2.2 do conv
        feature_conv_dec_minimap = F.relu(self.dec_conv_mini(feature_conv_dec_minimap))
        feature_conv_dec_screen = F.relu(self.dec_conv_screen(feature_conv_dec_screen))

        # 3.2.3 do deconv
        feature_conv_dec_minimap = F.relu(self.dec_deconv_mini1(feature_conv_dec_minimap))
        feature_conv_dec_minimap = F.tanh(self.dec_deconv_mini2(feature_conv_dec_minimap)) * 10

        feature_conv_dec_screen = F.relu(self.dec_deconv_screen1(feature_conv_dec_screen))
        feature_conv_dec_screen = F.tanh(self.dec_deconv_screen2(feature_conv_dec_screen)) * 10

        assert(list(feature_conv_dec_minimap.size()[1:]) == [3, 64, 64])
        assert(list(feature_conv_dec_screen.size()[1:]) == [6, 84, 84])

        return feature_fc_decision_out, feature_conv_dec_screen, feature_conv_dec_minimap

    def cuda(self, device=None):
        super(scNet, self).cuda(device)
        self.FC_mem.cuda()
        self.CONV_mem.cuda()

    def get_hidden(self):
        save_dict = {}
        save_dict["FC_mem"] = self.FC_mem.get_hidden()
        # save_dict["CONV_mem"] = self.CONV_mem.get_hidden()

        return save_dict

    def dump_hidden(self, hidden_dict):
        self.FC_mem.dump_hidden(hidden_dict["FC_mem"])
        # self.CONV_mem.dump_hidden(hidden_dict["CONV_mem"])
        pass

def scNetOutput2candidateAction(model_output, random_pick=False):
    out_fc, out_screen, out_minimap = model_output
    FUNCTIONS = actions.FUNCTIONS

    # backup torch one's
    out_fc_torch = out_fc
    out_screen_torch = out_screen
    out_minimap_torch = out_minimap

    out_fc = out_fc.cpu().detach().numpy()
    out_screen = out_screen.cpu().detach().numpy()
    out_minimap = out_minimap.cpu().detach().numpy()

    candidate_action = [0, 1, 3, 477, 12,
                        13, 331, 332, 274, 198,
                        199, 2]
    action_num = len(candidate_action)
    Q_value = [None for _ in range(action_num)]
    ret_action = [None for _ in range(action_num)]

    # id = 0 / no operator
    Q_value[0] = out_fc_torch[0, 0]
    ret_action[0] = actions.FunctionCall(0, [])

    # id = 1 / move camera
    candidate_2darr = out_minimap[0, 0]
    if random_pick:
        index = (random.randint(0, 63), random.randint(0, 63))
    else:
        index = np.unravel_index(candidate_2darr.argmax(), candidate_2darr.shape)
    max_2d = out_minimap_torch[0, 0][index]
    Q_value[1] = out_fc_torch[0, 1] * max_2d
    p = list(index)
    padding = 64 // 3 + 2
    p[0] = float(p[0]) / 64 * (64 - 2 * padding) + padding
    p[1] = float(p[1]) / 64 * (64 - 2 * padding) + padding
    ret_action[1] = actions.FunctionCall(1, [p])

    # id = 3 / select rectangle
    # Q value = choice priority * (candidate_triangle_prority + flag_priority)
    # candidate_triangle_proirty = p1_priorty + p2_priority (from out_screen[0, 0], [0, 1])
    # flag_prority from out_fc[0, 3], [0, 4], choice priorty from out_fc[0, 2]
    Q_value[2] = 0
    candidate_2darr = out_screen[0, 0]
    if random_pick:
        p1 = (random.randint(0, 83), random.randint(0, 83))
    else:
        p1 = np.unravel_index(candidate_2darr.argmax(), candidate_2darr.shape)
    Q_value[2] += out_screen_torch[0, 0, p1[0], p1[1]]
    # only take those rectangle not overflow the screen
    candidate_2darr = out_screen[0, 1][:84 - p1[0], :84 - p1[1]]
    if random_pick:
        p2 = (random.randint(0, candidate_2darr.shape[0] - 1), random.randint(0, candidate_2darr.shape[1] - 1))
    else:
        p2 = np.unravel_index(candidate_2darr.argmax(), candidate_2darr.shape)
    Q_value[2] += out_screen_torch[0, 1][:84 - p1[0], :84 - p1[1]][p2]
    if random_pick:
        i = random.randint(0, 1)
        Q_value[2] += out_fc_torch[0, 3 + i]
        flag = not bool(i)
    else:
        Q_value[2] += torch.max(out_fc_torch[0, 3], out_fc_torch[0, 4])
        flag = out_fc[0, 3] > out_fc[0, 4]
    Q_value[2] *= out_fc_torch[0, 2]
    # transform into candidate rectangle
    p1 = list(p1)
    p2 = list(p2)
    p2[0] += p1[0]
    p2[1] += p1[1]

    ret_action[2] = actions.FunctionCall(3, [[flag], p1, p2])

    # id = 477 / train marine
    Q_value[3] = out_fc_torch[0, 5]
    ret_action[3] = actions.FunctionCall(477, [[True]])

    # id = 12 / attack screen
    Q_value[4] = 0
    candidate_2darr = out_screen[0, 2]
    if random_pick:
        index = (random.randint(0, candidate_2darr.shape[0] - 1), random.randint(0, candidate_2darr.shape[1] - 1))
    else:
        index = np.unravel_index(candidate_2darr.argmax(), candidate_2darr.shape)
    max_2d = out_screen_torch[0, 2][index]
    Q_value[4] += max_2d
    p = list(index)
    Q_value[4] *= out_fc_torch[0, 6]
    ret_action[4] = FUNCTIONS.Attack_screen("now", p)

    # id = 13 / attack minimap
    Q_value[5] = 0
    candidate_2darr = out_minimap[0, 1]
    if random_pick:
        index = (random.randint(0, candidate_2darr.shape[0] - 1), random.randint(0, candidate_2darr.shape[1] - 1))
    else:
        index = np.unravel_index(candidate_2darr.argmax(), candidate_2darr.shape)
    max_2d = out_minimap_torch[0, 1][index]
    Q_value[5] += max_2d
    p = list(index)
    if random_pick:
        i = random.randint(0, 1)
        Q_value[5] += out_fc_torch[0, 8 + i]
        flag = not bool(i)
    else:
        Q_value[5] += torch.max(out_fc_torch[0, 8], out_fc_torch[0, 9])
        flag = out_fc[0, 8] > out_fc[0, 9]
    Q_value[5] *= out_fc_torch[0, 7]
    ret_action[5] = actions.FunctionCall(13, [[flag], p])

    # id = 331 / move screen
    Q_value[6] = 0
    candidate_2darr = out_screen[0, 3]
    if random_pick:
        index = (random.randint(0, candidate_2darr.shape[0] - 1), random.randint(0, candidate_2darr.shape[1] - 1))
    else:
        index = np.unravel_index(candidate_2darr.argmax(), candidate_2darr.shape)
    max_2d = out_screen_torch[0, 3][index]
    Q_value[6] += max_2d
    if random_pick:
        i = random.randint(0, 1)
        Q_value[6] += out_fc_torch[0, 11 + i]
        flag = not bool(i)
    else:
        Q_value[6] += torch.max(out_fc_torch[0, 11], out_fc_torch[0, 12])
        flag = out_fc[0, 11] > out_fc[0, 12]
    Q_value[6] *= out_fc_torch[0, 10]
    p = list(index)
    ret_action[6] = actions.FunctionCall(331, [[flag], p])

    # id = 332 / move minimap
    Q_value[7] = 0
    candidate_2darr = out_minimap[0, 2]
    if random_pick:
        index = (random.randint(0, candidate_2darr.shape[0] - 1), random.randint(0, candidate_2darr.shape[1] - 1))
    else:
        index = np.unravel_index(candidate_2darr.argmax(), candidate_2darr.shape)
    max_2d = out_minimap_torch[0, 2][index]
    Q_value[7] += max_2d
    if random_pick:
        i = random.randint(0, 1)
        Q_value[7] += out_fc_torch[0, 14 + i]
        flag = not bool(i)
    else:
        Q_value[7] += torch.max(out_fc_torch[0, 14], out_fc_torch[0, 15])
        flag = out_fc[0, 14] > out_fc[0, 15]
    Q_value[7] *= out_fc_torch[0, 13]
    p = list(index)
    ret_action[7] = actions.FunctionCall(332, [[flag], p])

    # id = 274 / hold position
    Q_value[8] = 0
    Q_value[8] += torch.max(out_fc_torch[0, 17], out_fc_torch[0, 18])
    flag = out_fc[0, 17] > out_fc[0, 18]
    Q_value[8] *= out_fc_torch[0, 16]
    ret_action[8] = actions.FunctionCall(274, [[flag]])

    # id = 198 / effect heal screen
    Q_value[9] = 0
    candidate_2darr = out_screen[0, 4]
    if random_pick:
        index = (random.randint(0, candidate_2darr.shape[0] - 1), random.randint(0, candidate_2darr.shape[1] - 1))
    else:
        index = np.unravel_index(candidate_2darr.argmax(), candidate_2darr.shape)
    max_2d = out_screen_torch[0, 4][index]
    Q_value[9] += max_2d
    if random_pick:
        i = random.randint(0, 1)
        Q_value[9] += out_fc_torch[0, 20 + i]
        flag = not bool(i)
    else:
        Q_value[9] += torch.max(out_fc_torch[0, 20], out_fc_torch[0, 21])
        flag = out_fc[0, 20] > out_fc[0, 21]
    Q_value[9] *= out_fc_torch[0, 19]
    p = list(index)
    ret_action[9] = actions.FunctionCall(198, [[flag], p])

    # id = 199 / effect heal autocast
    Q_value[10] = out_fc_torch[0, 22]
    ret_action[10] = actions.FunctionCall(199, [])

    # id = 2 / select_point
    Q_value[11] = 0
    candidate_2darr = out_screen[0, 5]
    if random_pick:
        index = (random.randint(0, candidate_2darr.shape[0] - 1), random.randint(0, candidate_2darr.shape[1] - 1))
    else:
        index = np.unravel_index(candidate_2darr.argmax(), candidate_2darr.shape)
    max_2d = out_screen_torch[0, 5][index]
    Q_value[11] += max_2d
    p = list(index)

    candidate_flags = out_fc[0, 24: 28]
    if random_pick:
        index = random.randint(0, len(candidate_flags) - 1)
    else:
        index = np.argmax(candidate_flags)
    max_val = out_fc_torch[0, 24: 28][index]
    Q_value[11] += max_val
    param = index
    ret_action[11] = actions.FunctionCall(2, [[param],p])

    candidate_action = np.array(candidate_action)
    ret_list = []
    for i in range(len(candidate_action)):
        # if candidate_action[i] in [1]:
        #     continue
        tmp_dict = {"Q_value": Q_value[i],
                    "action_id": candidate_action[i],
                    "action": ret_action[i]
                    }
        ret_list.append(tmp_dict)
    ret_list.sort(key=lambda x: x["Q_value"].cpu().detach(), reverse=True)
    return ret_list
if __name__ == "__main__":
    # basic sanity test
    # feature_fc = torch.randn((1, 63))
    # feature_screen = torch.randn((1, 8, 84, 84))
    # feature_minimap = torch.randn((1, 3, 64, 64))
    # model = scNet()
    # out_fc, out_screen, out_minimap = model(feature_fc, feature_screen, feature_minimap)
    #
    # print(out_fc.size(), out_screen.size(), out_minimap.size())

    # # cuda test
    # feature_fc = torch.randn((1, 63)).cuda()
    # feature_screen = torch.randn((1, 8, 84, 84)).cuda()
    # feature_minimap = torch.randn((1, 3, 64, 64)).cuda()
    # model = scNet()
    # model.cuda()
    # model_output = model(feature_fc, feature_screen, feature_minimap)
    # scNetOutput2candidateAction(model_output)

    # robust interation test
    model = scNet()
    model.cuda()
    for s in range(1000):
        feature_fc = torch.randn((1, 63)).cuda()
        feature_screen = torch.randn((1, 8, 84, 84)).cuda()
        feature_minimap = torch.randn((1, 3, 64, 64)).cuda()
        model = scNet()
        model.cuda()
        out_fc, out_screen, out_minimap = model(feature_fc, feature_screen, feature_minimap)
        model.dump_hidden(model.get_hidden())

        print(out_fc.size(), out_screen.size(), out_minimap.size())
