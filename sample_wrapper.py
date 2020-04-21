from torch.utils.data import Dataset
import torch
import pandas
import os

import dodge_parallel


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


class SampleWrapper:
    def __init__(self, game: dodge_parallel.GameParallel,
                 nn_time_step=60):

        self.game = game

        self.nn_time_step = nn_time_step

        self.game_stamp = 0
        self.overwrite = False

        self.memory = MemoryCSV()

    def run(self, actor, max_time, speed=1., overwrite=True, filename="./data.csv", console=True):

        if console:
            print("playing game ...", end="")

        self.memory.clear()

        self.game_stamp = 0  # time running, +1 every nn-tick

        self.game.obs = self.game.initial_obs()

        run = True
        total_time = 0

        while run:
            time_elapsed = self.nn_time_step
            total_time += time_elapsed

            run = total_time < max_time
            if console and total_time % int(max_time/10) < time_elapsed:
                print(" {}% ...".format(int(10 * total_time / int(max_time/10))), end="")

            self.update(time_elapsed, speed, actor)

        self.memory.finalize(filename=filename, overwrite=overwrite)

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
    def __init__(self, name="memory"):
        self.filename_stream = "./temp/{}_stream.csv.temp".format(name)

    def clear(self):
        # clear files
        with open(self.filename_stream, 'w'):
            pass

    def append(self, value):
        obs, action, obs_next, reward = value

        data = torch.cat([obs, action, obs_next, reward.float()], dim=1).numpy()
        pandas.DataFrame(data).to_csv(self.filename_stream, mode='a', index=None, header=False)

    # returns the number of rewards received
    def finalize(self, filename="./data.csv", overwrite=True, delete_temp=True):
        stream = pandas.read_csv(self.filename_stream, header=None).iloc[:, :].values

        pandas.DataFrame(stream).to_csv(filename, mode='w' if overwrite else 'a', index=None, header=False)
        if delete_temp:
            os.remove(self.filename_stream)