import pickle
import os

class recorder():

    def __init__(self):
        self.path = r"D:\workplace\pysc\cache\logs"
        self.filename = r"self-game-logs.pkl"
        self.full_path = os.path.join(self.path, self.filename)
        if os.path.exists(self.full_path):
            self.load()
        else:
            self.eposides = []
            self.rewards = []

    def dump(self, eposide, reward):
        self.eposides.append(eposide)
        self.rewards.append(reward)
        with open(self.full_path, "wb") as f:
            pickle.dump([self.eposides, self.rewards], f)

    def load(self):
        with open(self.full_path, "rb") as f:
            self.eposides, self.rewards = pickle.load(f)

if __name__ == "__main__":
    r = recorder()
    # r.dump(1, 100)
    r.load()
    print(r.eposides, r.rewards)
