import numpy
import torch
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import time

import dodge_game
import abstract_game
import rl_net


def load_data(filename="./data.csv", max_size=float('inf'), console=True):
    data_set = abstract_game.CustomDataSet.from_csv((30, 2, 30, 2, 2), filename=filename)

    size = int(min(max_size, len(data_set)))

    len_return = size
    len_rest = len(data_set) - size
    return_set, _ = random_split(data_set, [len_return, len_rest])

    if console:
        print("loaded dataset with {} elements".format(len_return))

    return return_set


def sample_game(actorNN, time_sec, overwrite=True, console=True):
    g = dodge_game.DodgeGame()
    rewards = g.run(actorNN, max_time=time_sec*1000, speed=1, run_in_background=True, overwrite=overwrite, console=console)
    return rewards


# 60 secs = 1000 datapoints
def create_data(actorNN, secs=30, times=1, max_size=float('inf'), overwrite=True, console=True, console_sample=True):
    rewards = sample_game(actorNN, secs, overwrite=overwrite, console=console_sample)
    n = 0

    if console:
        print("playing: ", end='')

    for i in range(times - 1):
        rewards += sample_game(actorNN, secs, overwrite=False, console=console_sample)
        if console and i > float(times-1)/10 * (n+1):
            n += 1
            print("{:.1f} % ... ".format(float(i) / (times-1) * 100), end='')

    if console:
        print("\nsampling done, gathered {} rewards per minute".format(rewards * 60 / secs / times))

    data_set = load_data(max_size=max_size, console=console)
    return data_set, rewards


def run(actorNN, speed=.5, noise=0, time_sec=100, track_events=False, console=False, initial_obs=None):
    old_noise = actorNN.noise
    actorNN.noise = noise
    g = dodge_game.DodgeGame()
    g.run(actorNN, speed=speed, track_events=track_events, max_time=time_sec * 1000, console=console,
          initial_obs=initial_obs, start_paused=initial_obs is not None)
    actorNN.noise = old_noise


def play(speed=.5, time_sec=100):
    g = dodge_game.DodgeGame()
    g.run(dodge_game.ActorHuman(), speed=speed, track_events=False, max_time=time_sec * 1000, console=False)


def load(agent, path='./agent.pth'):
    agent.load_state_dict(torch.load(path))
    agent.eval()

    return agent


def save(agent, path='./agent.pth'):
    torch.save(agent.state_dict(), path)


def train(agent, train_set, test_set, epochs=1, plot=True, rewards=(-1, 5), batch_size_train=1, batch_size_test=1, path='./agent.pth'):
    rl_net.train(epochs, agent, train_set, test_set, rewards, lr_rew=lr_rew, lr_next=lr_world, lr_pol=lr_pol,
                 plot=plot, c1=c1, c2=c2, c3=c3, batch_size_train=batch_size_train,
                 batch_size_test=batch_size_test, des_loss=des_loss, zero_step=zero_step)
    save(agent, path)


def loop(agent, actorNN, n=1, secs=30, times=40, epochs=10, rewards=(-1, 5), save_path='./agent.pth'):
    rew_count_list = []
    for i in range(n):
        start_time = time.time()

        train_set, test_set, reward_count = create_data(actorNN, secs=secs, times=times, console=True)

        train(agent, train_set, test_set, epochs=epochs, plot=plot_training, lr_rew=lr_rew, lr_world=lr_world, lr_pol=lr_pol,
              c1=c1, c2=c2, c3=c3, rewards=rewards, path=save_path, des_loss=des_loss, zero_step=zero_step,
              batch_size_train=16, batch_size_test=16)

        print("loop {} completed, took {:.1f} minutes, saving model to {}".format(i+1, (time.time()-start_time)/60., save_path))
        rew_count_list.append(reward_count)
        save(agent, save_path)

    if plot_rewards:
        results = numpy.array(rew_count_list)
        plt.scatter(range(n), results[:, 0], color='r', s=10, marker='o')
        plt.xlabel("Loops")
        plt.ylabel("Obstacles Hit")
        plt.figure()
        plt.scatter(range(n), results[:, 1], color='g', s=10, marker='o')
        plt.xlabel("Loops")
        plt.ylabel("Presents Collected")


