import torch
from torch.utils.data.dataset import random_split

import game_parallel
import wrapper_sample


def load_data(filename="./data.csv", max_size=float('inf'), console=True):
    data_set = wrapper_sample.DodgeDataset.from_csv((30, 2, 30, 2), filename=filename)

    size = int(min(max_size, len(data_set)))

    len_return = size
    len_rest = len(data_set) - size
    return_set, _ = random_split(data_set, [len_return, len_rest])

    if console:
        print("loaded dataset with {} elements".format(len_return))

    return return_set


def sample_game(actor, time_sec, n_parallel, overwrite=True, console=True, filename="./data.csv"):
    game = game_parallel.DodgeParallel(n_parallel)
    wrap = wrapper_sample.WrapperSample(game)
    wrap.run(actor, time_sec * 1000, overwrite=overwrite, console=console, filename=filename)


# 60 secs = 1000 datapoints
def create_data(actor, secs=30, times=1, n_parallel=1, max_size=float('inf'), overwrite=True, console=True,
                console_sample=False, filename="./data.csv"):

    if console:
        print("playing: ", end='')

    sample_game(actor, secs, n_parallel, overwrite=overwrite, console=console_sample, filename=filename)
    n = 0

    for i in range(times - 1):
        sample_game(actor, secs, n_parallel, overwrite=False, console=console_sample, filename=filename)
        if console and i + 1 >= float(times-1)/10 * (n+1):
            n += 1
            print("{:.0f} % ... ".format(float(i) / (times-1) * 100), end='')

    if console:
        print("\nsampling done")

    data_set = load_data(max_size=max_size, console=console, filename=filename)
    rewards = data_set.dataset.reward.sum(dim=0)

    return data_set, rewards


def load(agent, path='./agent.pth', device='cpu'):
    agent.load_state_dict(torch.load(path, map_location=torch.device(device)))
    agent.eval()

    return agent


def save(agent, path='./agent.pth'):
    torch.save(agent.state_dict(), path)
