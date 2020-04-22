from torch.utils.data import Dataset
import torch
import pandas

import game_parallel


class WrapperSample:
    def __init__(self, game: game_parallel.GameParallel,
                 millis_per_tick=60):

        self.game = game

        self.millis_per_tick = millis_per_tick

        self.game_stamp = 0
        self.overwrite = False

        self.memory = None

    def run(self, actor, max_time, speed=1., overwrite=True, filename="./data.csv", console=True):

        if console:
            print("playing game ...", end="")

        self.memory = MemoryCSV(filename)
        if overwrite:
            self.memory.clear()

        self.game_stamp = 0  # time running, +1 every nn-tick

        self.game.obs = self.game.initial_obs()

        run = True
        total_time = 0

        while run:
            time_elapsed = self.millis_per_tick
            total_time += time_elapsed

            run = total_time < max_time
            if console and total_time % int(max_time/10) < time_elapsed:
                print(" {}% ...".format(int(10 * total_time / int(max_time/10))), end="")

            self.update(time_elapsed, speed, actor)

        if console:
            print(" finalized")

    def update(self, time_elapsed, speed, actor):

        obs = self.game.obs
        reward = self.game.rewards()

        action = actor.get_action(self.game.obs, None, True)

        self.game.action_move(action, time_elapsed, speed)
        self.game.action_triggered()

        # adding last time step to memory, and advancing game_stamp by 1
        obs_next = self.game.obs

        # writing state of the game to temporary csv
        self.memory.append((obs, action, obs_next, reward))
        self.game_stamp += 1


class MemoryCSV:
    def __init__(self, filename):
        self.filename = filename

    def clear(self):
        # clear files
        with open(self.filename, 'w'):
            pass

    def append(self, value):
        obs, action, obs_next, reward = value

        data = torch.cat([obs, action, obs_next, reward.float()], dim=1).numpy()
        pandas.DataFrame(data).to_csv(self.filename, mode='a', index=None, header=False)


class DodgeDataset(Dataset):
    def __init__(self, obs, action, obs_next, reward):
        self.obs = obs
        self.action = action
        self.obs_next = obs_next
        self.reward = reward

    def __getitem__(self, i):
        return self.obs[i], self.action[i], self.obs_next[i], self.reward[i]

    def __len__(self):
        return self.obs.shape[0]

    def to_csv(self, filename="./data.csv", overwrite=False):
        data = torch.cat([self.obs, self.action, self.obs_next, self.reward], dim=1).numpy()

        data_frame = pandas.DataFrame(data=data, index=None, columns=None)

        data_frame.to_csv(filename, mode='w' if overwrite else 'a', index=None, header=False)

    @classmethod
    def from_csv(cls, split, filename="./data.csv"):
        n_obs, n_act, n_obs_next, n_rew = split
        data_frame = pandas.read_csv(filename, header=None)

        obs = torch.FloatTensor().new_tensor(data_frame.iloc[:,
                                             :n_obs].values)
        act = torch.FloatTensor().new_tensor(data_frame.iloc[:,
                                             n_obs:
                                             n_obs + n_act].values)
        onx = torch.FloatTensor().new_tensor(data_frame.iloc[:,
                                             n_obs + n_act:
                                             n_obs + n_act + n_obs_next].values)
        rew = torch.FloatTensor().new_tensor(data_frame.iloc[:,
                                             n_obs + n_act + n_obs_next:
                                             n_obs + n_act + n_obs_next + n_rew].values)

        return cls(obs, act, onx, rew)
