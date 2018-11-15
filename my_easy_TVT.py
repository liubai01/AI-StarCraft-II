"""
This script runs the contest on Simple64 Map w.r.t agent_Alpha and built-in
very easy bot. The speed is set to x1 and both race are set to terran.
"""
import os

def list2str(cmd):
    """
    connect the string in list with space
    :param cmd: a list contains the command and flag
    :return: a string
    """
    ret = ""
    for i in range(len(cmd) - 1):
        ret += cmd[i] + " "
    ret += cmd[-1]
    print(ret)
    return ret

def exec_cmd(cmd):
    """
    operate the list-like command:
    cmd: [<command>, <flag1_key>, <flag1_val>, <flag2_key>, <flag2_val>, ...]
    :param cmd: a list looked like above
    :return: the command return code, 0 if succeed
    """
    return os.system(list2str(cmd))


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
