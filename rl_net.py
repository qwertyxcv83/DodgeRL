import torch
import torch.nn.functional as functional

import networks


class ModelAgent(torch.nn.Module):
    def __init__(self, n_obs, n_reward, n_act, cuda):
        super(ModelAgent, self).__init__()
        n_hidden = 100
        self.n_obs = n_obs
        self.n_reward = n_reward
        self.n_act = n_act
        self.is_cuda = cuda

        self.inp = torch.nn.Linear(n_obs + n_act, n_hidden)
        self.layer = networks.DenseNet([networks.MaxLayer(n_hidden, n_hidden, n_max=3) for _ in range(4)])
        self.out = torch.nn.Linear(n_hidden, n_reward + n_reward + n_act + n_obs)

    def forward(self, values):
        obs, act = values
        if act is None:
            act = torch.FloatTensor().new_zeros(obs.shape[0], self.n_act)
        if self.is_cuda:
            obs = obs.cuda()
            act = act.cuda()

        x = self.inp(torch.cat([obs, act], dim=1))
        x = self.layer(x)
        out = self.out(x)

        reward = out[:, :self.n_reward].sigmoid()
        estimation = out[:, self.n_reward:self.n_reward * 2].abs()
        policy = out[:, self.n_reward * 2:self.n_reward * 2 + self.n_act].tanh()
        world = out[:, -self.n_obs:]

        return reward, estimation, policy, world

    def loss(self, data,
             weights=torch.FloatTensor().new_tensor([4, 1, 1, 10, 1]),
             reward_weights=torch.FloatTensor().new_tensor([-1, 1])):
        obs_in, act_in, obs_next_in, reward_in = data
        if self.is_cuda:
            obs_in = obs_in.cuda()
            act_in = act_in.cuda()
            obs_next_in = obs_next_in.cuda()
            reward_in = reward_in.cuda()
            weights = weights.cuda()

        reward, estimation, policy, _ = self((obs_in, act_in))
        _, e_next, _, _ = self((obs_next_in, policy))

        loss_reward = functional.binary_cross_entropy(reward, reward_in)
        loss_estimation = ModelAgent.estimator_loss(estimation, e_next, reward_in)
        loss_policy = ModelAgent.policy_loss(estimation, e_next, reward_weights)

        loss = torch.cat([loss_reward.flatten(),
                          loss_estimation[0].flatten(),
                          loss_estimation[1].flatten(),
                          loss_estimation[2].flatten(),
                          loss_policy.flatten()],
                         dim=0) * weights

        return loss

    @staticmethod
    def estimator_loss(estimation, e_next, reward_in):
        # batch mean should be around 1
        loss_mean = (estimation.mean(dim=0) - 1).mean() ** 2

        # est > 0, reward_in = 1/0 --> mean distance to zero for reward_in=True squared
        loss_one = ((estimation * reward_in).sum(dim=0) / reward_in.sum(dim=0).clamp(1)).mean() ** 2

        # difference to next estimation
        loss_difference = (estimation - e_next).mean() ** 2

        return loss_mean, loss_one, loss_difference

    @staticmethod
    def policy_loss(estimation, e_next, reward_weights):
        return ((estimation - e_next) * reward_weights).mean()

    def reward_accuracy(self, data):
        with torch.no_grad():
            obs_in, act_in, obs_next_in, reward_bool_in = data
            if self.is_cuda:
                obs_in = obs_in.cuda()
                act_in = act_in.cuda()
                obs_next_in = obs_next_in.cuda()
                reward_bool_in = reward_bool_in.cuda()

            pred = self((obs_in, None))[0] > .5
            truth = reward_bool_in.bool()

            correct_one = (pred & truth).sum(dim=0).cpu()
            total_one = truth.sum(dim=0).cpu()
            correct_zero = (~pred & ~truth).sum(dim=0).cpu()
            total_zero = (~truth).sum(dim=0).cpu()

            return correct_one, total_one, correct_zero, total_zero

    def reward_gradient(self, obs, rewards):
        if self.is_cuda:
            obs = obs.cuda()
            rewards = rewards.cuda()

        obs.requires_grad_()
        if obs.grad is not None:
            obs.grad.zero_()

        rew = self((obs, None))[0] * rewards.reshape(1, 2)
        rew.sum().backward()

        grad = obs.grad
        obs.requires_grad_(False)
        return grad
