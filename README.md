# AI-StarCraft-II
![](https://github.com/liubai01/AI-StarCraft-II/blob/master/img/header.png)

Final project of CS181 in ShanghaiTech. Focus on experiments with different opponent setting policy with basis of DQN method on StarCraft 2.

**Teammate:** Keyi-Yuan, Qin-QI, Ruiqi-Liu, Yintao-Xu

For the reason that we do slight modification to the pysc2, i include pysc2 lib in this repository([pysc2](https://github.com/deepmind/pysc2)) under Apache License 2.0. 

### **Quick Start**

1. git clone https://github.com/liubai01/AI-StarCraft-II.git
2. Copy `map/SimpleContest.SC2Map` to your map directory. (e.g: on my PC, it is: `D:\billizard\StarCraft II\Maps\Melee`, it depends on where you install the game)
3. IMPORTANT: Modify the log file path in `lib\pysc2_info_saver.py`, the recorder class! （留言：帮我跑程序的话一定要设置这个，否则实验白做了。。）

### Documents

TBD, for now, there are only some scratches

1. [action space](https://github.com/liubai01/AI-StarCraft-II/blob/master/documents/action_space.md)

2. [feature engineering-observation](https://github.com/liubai01/AI-StarCraft-II/blob/master/documents/observation.md)

Other scratches: https://github.com/Q71998/yascai

### Reference documentation

General idea of pysc2: [[Portal](https://github.com/deepmind/pysc2)]

Environment in pysc2: [[Portal](https://github.com/deepmind/pysc2/blob/master/docs/environment.md)]

### **Update Log**

2018/11/15: backbone of this project. Realize detection the center of build(currently command center) by `lib/building/get_building_center(<obs>, <uid>)`. Realize basic agent in agent_Alpha.

2018/12/19: backbone of the self-game RL learning setting