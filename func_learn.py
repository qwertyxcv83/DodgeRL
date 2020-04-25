import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

import game_parallel
import wrapper_sample


# split: obs, act, n_obs, rew
def load_data(split, filename="./data.csv", max_size=float('inf'), console=True):
    data_set = wrapper_sample.CustomDataset.from_csv(split, filename=filename)

    size = int(min(max_size, len(data_set)))

    len_return = size
    len_rest = len(data_set) - size
    return_set, _ = random_split(data_set, [len_return, len_rest])

    if console:
        print("loaded dataset with {} elements".format(len_return))

    return return_set


# data_set class should be torch.utils.data.dataset.SubSet
def reduce_dataset(model_agent, data_set, x, filename="./data.csv", batch_size=1024,
                   weights=torch.FloatTensor().new_tensor([1] * 5)):
    if model_agent.is_cuda:
        weights = weights.cuda()
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, drop_last=False)
    losses = torch.FloatTensor().new_zeros(len(data_set))
    with torch.no_grad():
        model_agent.eval()
        pos = 0
        for data in data_loader:
            loss = (model_agent.loss(data) * weights).sum(dim=1).cpu()
            size = loss.shape[0]
            losses[pos:pos+size] = loss
            pos += size
    print("losses calculated, mean: {:.4f}".format(losses.mean()), end="")
    top = torch.topk(losses, int(x * len(data_set)))
    print(", top {}% loss mean: {:.4f}".format(x * 100, top[0].mean()))
    indices = torch.Tensor().new_tensor(data_set.indices)[top[1], ]
    wrapper_sample.CustomDataset.from_tuple(data_set.dataset[indices.long(), ]).to_csv(filename, overwrite=True)
    print("done, saved at {}".format(filename))


def sample_game(actor, time_sec, n_parallel, game_class, overwrite=True, console=True, filename="./data.csv"):
    game = game_class(n_parallel)
    wrap = wrapper_sample.WrapperSample(game)
    wrap.run(actor, time_sec * 1000, overwrite=overwrite, console=console, filename=filename)


# 60 secs = 1000 datapoints
# split: (obs, act, next_obs, rew)
def create_data(actor, split, secs=30, times=1, n_parallel=1, max_size=float('inf'), overwrite=True, console=True,
                console_sample=False, filename="./data.csv", game_class=game_parallel.DodgeParallel):

    if console:
        print("playing: ", end='')

    sample_game(actor, secs, n_parallel, game_class, overwrite=overwrite, console=console_sample, filename=filename)
    n = 0

    for i in range(times - 1):
        sample_game(actor, secs, n_parallel, game_class, overwrite=False, console=console_sample, filename=filename)
        if console and i + 1 >= float(times-1)/10 * (n+1):
            n += 1
            print("{:.0f} % ... ".format(float(i) / (times-1) * 100), end='')

    if console:
        print("sampling done")

    data_set = load_data(split, max_size=max_size, console=console, filename=filename)

    return data_set


def load(agent, path='./agent.pth', device='cpu'):
    agent.load_state_dict(torch.load(path, map_location=torch.device(device)))
    agent.eval()
    print("model loaded from {} to device {}".format(path, device))
    return agent


def save(agent, path='./agent.pth'):
    torch.save(agent.state_dict(), path)
    print("model saved to {}".format(path))
