import pygame
import torch
from torch.utils.data import Dataset
import pandas
import numpy
import os


class UserInput:
    def __init__(self):
        self.run = True
        self.up = False
        self.down = False
        self.right = False
        self.left = False
        self.space = False
        self.mouse_left = False  # only true if pressed this tick
        self.mouse_right = False  # only true if pressed this tick
        self.mouse_wheel = False  # only true if pressed this tick
        self.mouse_x = 0
        self.mouse_y = 0

    def set_user_input(self, events):
        keydown = False
        for event in events:
            if event.type == pygame.QUIT:
                self.run = False
            if event.type == pygame.KEYDOWN:
                keydown = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_left, self.mouse_right, self.mouse_wheel = pygame.mouse.get_pressed()
            else:
                self.mouse_left = False
                self.mouse_right = False
                self.mouse_wheel = False

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            self.up = True
            self.down = False
            self.right = False
            self.left = False
        if pressed[pygame.K_DOWN]:
            self.up = False
            self.down = True
            self.right = False
            self.left = False
        if pressed[pygame.K_RIGHT]:
            self.up = False
            self.down = False
            self.right = True
            self.left = False
        if pressed[pygame.K_LEFT]:
            self.up = False
            self.down = False
            self.right = False
            self.left = True
        if pressed[pygame.K_SPACE] and keydown:
            self.space = True
        else:
            self.space = False
        self.mouse_x, self.mouse_y = pygame.mouse.get_pos()

    def reset(self):
        self.run = True
        self.up = False
        self.down = False
        self.right = False
        self.left = False
        self.space = False
        self.mouse_left = False
        self.mouse_right = False
        self.mouse_wheel = False
        self.mouse_x = 0
        self.mouse_y = 0


class CustomDataSet(Dataset):
    def __init__(self, obs, action, obs_next, reward_time, reward):
        self.obs = obs
        self.action = action
        self.obs_next = obs_next
        self.reward_time = reward_time
        self.reward = reward

    def __getitem__(self, i):
        return self.obs[i], self.action[i], self.obs_next[i], self.reward_time[i], self.reward[i]

    def __len__(self):
        return self.obs.shape[0]

    def to_csv(self, filename="./data.csv", overwrite=False):
        data = torch.cat([self.obs, self.action, self.obs_next, self.reward_time, self.reward], dim=1).numpy()

        data_frame = pandas.DataFrame(data=data, index=None, columns=None)

        data_frame.to_csv(filename, mode='w' if overwrite else 'a', index=None, header=False)

    @classmethod
    def from_csv(cls, split, filename="./data.csv"):
        n_obs, n_act, n_obs_next, n_rew_time, n_rew = split
        data_frame = pandas.read_csv(filename, header=None)

        obs = torch.FloatTensor().new_tensor(data_frame.iloc[:, :n_obs].values)
        act = torch.FloatTensor().new_tensor(data_frame.iloc[:, n_obs:n_obs + n_act].values)
        onx = torch.FloatTensor().new_tensor(data_frame.iloc[:, n_obs + n_act:n_obs + n_act + n_obs_next].values)
        rti = torch.FloatTensor().new_tensor(data_frame.iloc[:, n_obs + n_act + n_obs_next:n_obs + n_act + n_obs_next + n_rew_time].values)
        rew = torch.FloatTensor().new_tensor(data_frame.iloc[:, n_obs + n_act + n_obs_next + n_rew_time:n_obs + n_act + n_obs_next + n_rew_time + n_rew].values)

        return cls(obs, act, onx, rti, rew)


class AbstractGame:
    def __init__(self, n_rewards, observation_size, action_size, width, height, nn_time_step=60, frame_rate=121):
        self.frame_rate = frame_rate
        self.nn_time_step = nn_time_step
        self.observation_size = observation_size
        self.action_size = action_size
        self.n_rewards = n_rewards

        self.width = width
        self.height = height

        self.user_input = UserInput()
        self.nn_step_timer = 0
        self.game_stamp = 0
        self.pause = False
        self.overwrite = False

        # always a torch tensor
        self.observation = self.initial_obs()

        self.memory = MemoryCSV()

    def run(self, actor, max_time, speed, run_in_background, track_events, overwrite, filename="./data.csv",
            console=True, initial_obs=None, start_paused=False):
        if not run_in_background:
            window = pygame.display.set_mode((self.width, self.height))
            pygame.font.init()
            font = pygame.font.SysFont('Verdana', 20)
            clock = pygame.time.Clock()

        if console:
            print("playing game ...", end="")

        if track_events:
            self.memory.clear()

        self.user_input.reset()
        self.pause = start_paused
        self.game_stamp = 0  # time running, +1 every nn-tick
        self.nn_step_timer = 0  # sum up over time, next nn-tick when greater than threshold
        self.observation = self.initial_obs() if initial_obs is None else initial_obs

        run = True
        total_time = 0

        while run:
            if not run_in_background:
                time_elapsed = clock.tick(self.frame_rate)
                self.user_input.set_user_input(pygame.event.get())
            else:
                time_elapsed = self.nn_time_step
            total_time += time_elapsed

            self.pause = self.user_input.space ^ self.pause
            run = self.user_input.run and total_time < max_time
            if console and total_time % int(max_time/10) < time_elapsed:
                print(" {}% ...".format(int(10 * total_time / int(max_time/10))), end="")

            if not self.pause:
                self.update(time_elapsed, speed, actor, run_in_background, track_events)
            elif not run_in_background:
                self.update_pause(actor)

            if not run_in_background:
                self.draw(window, actor, font)
                if self.pause:
                    self.draw_pause(window, actor, font)

                pygame.display.update()

        if not run_in_background:
            pygame.quit()
        if console:
            print()

        if track_events:
            # calculating time until reward and writing to data.csv file
            return self.memory.finalize(filename=filename, overwrite=overwrite, console=console)

    def update(self, time_elapsed, speed, actor, run_in_background, track_events):
        is_nn_time_step = False

        if run_in_background:
            is_nn_time_step = True
        else:
            self.nn_step_timer += time_elapsed
            if self.nn_step_timer >= self.nn_time_step:
                is_nn_time_step = True
                self.nn_step_timer %= self.nn_time_step

        if is_nn_time_step and track_events:
            reward = self.receive_reward()
            if reward is not None and reward.any():
                # make a new reward entry
                self.memory.reward_received(self.game_stamp, reward)

        # step, certain events should only be triggered if it's a nn_time_step
        obs = self.observation
        action = actor.get_action(self.observation, self.user_input, is_nn_time_step)
        self.step_obs(time_elapsed, action, speed, is_nn_time_step)  # should create new obs object

        # adding last time step to memory, and advancing game_stamp by 1
        if is_nn_time_step and track_events:
            obs_next = self.observation

            # writing state of the game to temporary csv
            self.memory.append((obs, action, obs_next))
            self.game_stamp += 1

    def update_pause(self, actor):
        raise NotImplementedError

    def draw(self, window, actor, font):
        if actor.is_agent:
            with torch.no_grad():
                p = actor.agent.get_reward(self.observation)
            for i in range(p.shape[1]):
                text_surface = font.render('Reward: {:.4f}'.format(p.numpy()[0, i]), False, (0, 0, 0))
                window.blit(text_surface, (0, 50*i))

    # if the game is paused, additional info will be displayed
    def draw_pause(self, window, actor, font):
        raise NotImplementedError

    # returns tuple
    def decode_obs(self, obs):
        raise NotImplementedError

    # returns float tensor
    def encode_obs(self, values):
        raise NotImplementedError

    # return float tensor
    def initial_obs(self):
        raise NotImplementedError

    # returns None
    def step_obs(self, time_elapsed, action, speed, is_nn_time_step):
        raise NotImplementedError

    # returns bool tensor
    def receive_reward(self):
        raise NotImplementedError


class MemoryCSV:
    def __init__(self, name="memory"):
        self.filename_stream = "./temp/{}_stream.csv.temp".format(name)
        self.filename_reward = "./temp/{}_reward.csv.temp".format(name)

    def clear(self):
        # clear files
        with open(self.filename_stream, 'w'): pass
        with open(self.filename_reward, 'w'): pass

    def append(self, value):
        obs, action, obs_next = value

        data = torch.cat([obs, action, obs_next], dim=1).numpy()
        pandas.DataFrame(data).to_csv(self.filename_stream, mode='a', index=None, header=False)

    # reward is bool tensor, shape = [n_rew]
    def reward_received(self, game_stamp, reward):
        data = numpy.concatenate(([game_stamp], reward.numpy())).reshape(1, 1+reward.shape[0])

        pandas.DataFrame(data).to_csv(self.filename_reward, mode='a', index=None, header=False)

    # returns the number of rewards received
    def finalize(self, filename="./data.csv", overwrite=True, delete_temp=True, console=True):
        stream = pandas.read_csv(self.filename_stream, header=None).iloc[:, :].values
        reward_matrix = pandas.read_csv(self.filename_reward, header=None).iloc[:, :].values

        n_rews = reward_matrix.shape[1] - 1
        # time until next reward
        reward_times = numpy.zeros((stream.shape[0], n_rews))
        # if reward received this (nn-)tick
        reward_bool = numpy.zeros((stream.shape[0], n_rews))

        for i in range(n_rews):
            reduced_matrix = reward_matrix[:, (0, i+1)]  # reduced to one reward-type
            stream_counter = 0
            for event in reduced_matrix:
                if event[1]:
                    game_stamp = event[0]

                    reward_times[stream_counter:game_stamp, i] = range(game_stamp - stream_counter, 0, -1)
                    reward_bool[game_stamp, i] = 1

                    stream_counter = game_stamp

        data = numpy.concatenate((stream, reward_times, reward_bool), axis=1)

        pandas.DataFrame(data).to_csv(filename, mode='w' if overwrite else 'a', index=None, header=False)

        count_rewards = reward_matrix[:, 1:].sum(0)
        if console:
            print("saved gathered data, number of rewards achieved: {}".format(count_rewards))
        if delete_temp:
            os.remove(self.filename_stream)
            os.remove(self.filename_reward)

        return count_rewards


class Actor:
    def __init__(self, is_agent):
        self.is_agent = is_agent

    def get_action(self, observation, user_input, nn_step):
        raise NotImplementedError


class ActorNN(Actor):
    def __init__(self, agent, noise=.5, max_speed=1.5):
        super().__init__(True)
        self.agent = agent
        self.noise = .5#
        self.max_speed = max_speed
        self.act = torch.FloatTensor().new_zeros((1, agent.n_act))

    def get_action(self, observation, user_input, nn_step):
        if nn_step:
            # with # torch.no_grad():
            noise_normal = torch.randn(1, self.agent.n_act)
            act = self.agent.get_action(observation)

            self.act = act * (1 + self.noise * noise_normal) * self.max_speed

        return self.act
