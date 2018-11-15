import os

def list2str(cmd):
    ret = ""
    for i in range(len(cmd) - 1):
        ret += cmd[i] + " "
    ret += cmd[-1]
    print(ret)
    return ret

def exec_cmd(cmd):
    os.system(list2str(cmd))

if __name__ == "__main__":
    cmd = ["python"]
    cmd += ["-m", "pysc2.bin.agent"]
    cmd += ["--map", "Simple64"]
    cmd += ["--agent", "agent_Alpha.alphaAgent"]
    cmd += ["--agent2", "Bot"]
    cmd += ["--agent_race", "terran"]
    cmd += ["--agent2_race", "terran"]
    cmd += ["--max_episodes", "1"]
    cmd += ["--step_mul", "1"]
    exec_cmd(cmd)
