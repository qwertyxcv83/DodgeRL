import importlib
import numpy
import torch
from torch import FloatTensor

import torch.nn.functional as functional
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas
import time
import pygame

import dodge_game
import abstract_game
import util
import nn_util
import networks
import rl_net

import mnist_tester


actorNN, agent, test_set, train_set = None, None, None, None


def reload_imports():
    importlib.reload(dodge_game)
    importlib.reload(abstract_game)
    importlib.reload(util)
    importlib.reload(nn_util)
    importlib.reload(rl_net)
    importlib.reload(mnist_tester)
    importlib.reload(networks)


def load_data(filename="./data.csv", max_size=float('inf'), proportion=0.8, console=True):
    global train_set, test_set
    data_set = abstract_game.CustomDataSet.from_csv((30, 2, 30, 2, 2), filename=filename)

    size = int(min(max_size, len(data_set)))
    len_train = int(size * proportion)
    len_test = size - len_train
    len_rest = len(data_set) - size
    train_set, test_set, _ = random_split(data_set, [len_train, len_test, len_rest])

    if console:
        print("loaded train set with {} elements and test set with {} elements".format(len_train, len_test))

    return train_set, test_set


def sample_game(time_sec, overwrite=True, console=True):
    global actorNN
    g = dodge_game.DodgeGame()
    rewards = g.run(actorNN, max_time=time_sec*1000, speed=1, run_in_background=True, overwrite=overwrite, console=console)
    return rewards


# 60 secs = 800 train + 200 test
def create_data(secs=30, times=1, max_size=float('inf'), overwrite=True, console=True, console_sample=False, proportion=.8):
    global train_set, test_set

    rewards = sample_game(secs, overwrite=overwrite, console=console_sample)
    for _ in range(times - 1):
        rewards += sample_game(secs, overwrite=False, console=console_sample)

    if console:
        print("sampling done, gathered {} rewards per minute".format(rewards * 60 / secs / times))

    train_set, test_set = load_data(max_size=max_size, proportion=proportion, console=console)
    return train_set, test_set, rewards


def run(speed=.5, noise=0, time_sec=100, track_events=False, console=False, initial_obs=None):
    global actorNN
    old_noise = actorNN.noise
    actorNN.noise = noise
    g = dodge_game.DodgeGame()
    g.run(actorNN, speed=speed, track_events=track_events, max_time=time_sec * 1000, console=console,
          initial_obs=initial_obs, start_paused=initial_obs is not None)
    actorNN.noise = old_noise


def play(speed=.5, time_sec=100):
    g = dodge_game.DodgeGame()
    g.run(dodge_game.ActorHuman(), speed=speed, track_events=False, max_time=time_sec * 1000, console=False)


def load(path='./agent.pth'):
    global agent
    set_agent()

    agent.load_state_dict(torch.load(path))
    agent.eval()


def save(path='./agent.pth'):
    global agent
    torch.save(agent.state_dict(), path)


def train(epochs=1, lr_rew=.01, lr_world=.01, lr_pol=.01, plot=True, c1=1, c2=2, c3=.5,
          rewards=torch.FloatTensor().new_tensor([-1, 5]), batch_size_train=1, batch_size_test=1, des_loss=float('inf'),
          zero_step=False, path='./agent.pth'):
    global agent
    rl_net.train(epochs, agent, train_set, test_set, rewards, lr_rew=lr_rew, lr_next=lr_world, lr_pol=lr_pol,
                 plot=plot, c1=c1, c2=c2, c3=c3, batch_size_train=batch_size_train,
                 batch_size_test=batch_size_test, des_loss=des_loss, zero_step=zero_step)
    save(path)


def loop(n=1, secs=30, times=40, epochs=10, plot_training=False, plot_rewards=True, lr_rew=.01, lr_world=.1, lr_pol=.01,
         c1=1, c2=.5, c3=.5, rewards=torch.FloatTensor().new_tensor([-1, 5]), save_path='./agent.pth',
         des_loss=float('inf'), zero_step=False):
    global train_set, test_set

    rew_count_list = []
    for i in range(n):
        start_time = time.time()

        train_set, test_set, reward_count = create_data(secs=secs, times=times, console=True)

        train(epochs=epochs, plot=plot_training, lr_rew=lr_rew, lr_world=lr_world, lr_pol=lr_pol,
              c1=c1, c2=c2, c3=c3, rewards=rewards, path=save_path, des_loss=des_loss, zero_step=zero_step,
              batch_size_train=16, batch_size_test=16)

        print("loop {} completed, took {:.1f} minutes, saving model to {}".format(i+1, (time.time()-start_time)/60., save_path))
        rew_count_list.append(reward_count)
        save(save_path)

    if plot_rewards:
        results = numpy.array(rew_count_list)
        plt.scatter(range(n), results[:, 0], color='r', s=10, marker='o')
        plt.xlabel("Loops")
        plt.ylabel("Obstacles Hit")
        plt.figure()
        plt.scatter(range(n), results[:, 1], color='g', s=10, marker='o')
        plt.xlabel("Loops")
        plt.ylabel("Presents Collected")


def set_agent():
    global actorNN, agent
    agent = rl_net.ModelAgent(30, 2, 2)
    actorNN = abstract_game.ActorNN(agent)


set_agent()

load_data()

mnmodel = mnist_tester.Model()

#
