import torch
import torch.nn.functional as functional

import networks


class ModelAgent(torch.nn.Module):
    def __init__(self, n_in, n_reward, n_act, cuda):
        super(ModelAgent, self).__init__()
        n_hidden = 100
        self.n_obs = n_in
        self.n_reward = n_reward
        self.n_act = n_act
        self.is_cuda = cuda

        self.inp = torch.nn.Linear(n_in, n_hidden)
        self.layer = networks.DenseNet([networks.MaxLayer(n_hidden, n_hidden, n_max=3) for _ in range(4)])
        self.out = torch.nn.Linear(n_hidden, n_reward)

    def forward(self, obs):
        if self.is_cuda:
            obs = obs.cuda()

        x = self.inp(obs)
        x = self.layer(x)
        out = self.out(x)

        return out.sigmoid()

    def get_action(self, obs):
        if self.is_cuda:
            obs = obs.cuda()

        rew_grad = self.reward_gradient(obs, torch.FloatTensor().new_tensor([-1, .5]))
        return rew_grad[:, :2].tanh()
        # return torch.randn(2)
        # return torch.tanh(self.policy(self.hidden(obs)))

    def get_reward(self, obs):
        if self.is_cuda:
            obs = obs.cuda()

        return self(obs)

    def reward_gradient(self, obs, rewards):
        if self.is_cuda:
            obs = obs.cuda()
            rewards = rewards.cuda()

        obs.requires_grad_()
        if obs.grad is not None:
            obs.grad.zero_()

        rew = self.get_reward(obs) * rewards.reshape(1, 2)
        rew.sum().backward()

        grad = obs.grad
        obs.requires_grad_(False)
        return grad

    def loss(self, data):
        obs_in, act_in, obs_next_in, reward_time_in, reward_bool_in = data
        if self.is_cuda:
            obs_in = obs_in.cuda()
            act_in = act_in.cuda()
            obs_next_in = obs_next_in.cuda()
            reward_time_in = reward_time_in.cuda()
            reward_bool_in = reward_bool_in.cuda()

        reward = self.get_reward(obs_in)

        loss = functional.binary_cross_entropy(reward, reward_bool_in)

        return loss

    def reward_accuracy(self, data):
        obs_in, act_in, obs_next_in, reward_time_in, reward_bool_in = data
        if self.is_cuda:
            obs_in = obs_in.cuda()
            act_in = act_in.cuda()
            obs_next_in = obs_next_in.cuda()
            reward_time_in = reward_time_in.cuda()
            reward_bool_in = reward_bool_in.cuda()

        pred = self.get_reward(obs_in) > .5
        truth = reward_bool_in.bool()

        correct_one = (pred & truth).sum(dim=0)
        total_one = truth.sum(dim=0)
        correct_zero = (~pred & ~truth).sum(dim=0)
        total_zero = (~truth).sum(dim=0)

        return correct_one, total_one, correct_zero, total_zero
