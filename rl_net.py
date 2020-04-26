import torch
import torch.nn.functional as functional

import networks


class ModelAgent(torch.nn.Module):
    def __init__(self, n_obs, n_act, reward_weights, cuda, n_hidden=200):
        super(ModelAgent, self).__init__()
        self.n_obs = n_obs
        self.n_reward = len(reward_weights)
        self.reward_weights = torch.FloatTensor().new_tensor(reward_weights).cuda() if cuda else \
            torch.FloatTensor().new_tensor(reward_weights)
        self.n_act = n_act
        self.is_cuda = cuda

        self.inp = torch.nn.Linear(n_obs + n_act, n_hidden)
        self.layer = networks.DenseNet([networks.MaxLayer(n_hidden, n_hidden, n_max=3) for _ in range(4)])
        self.out = torch.nn.Linear(n_hidden, self.n_reward + self.n_reward + n_act + self.n_reward)

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
        estimation = out[:, self.n_reward:self.n_reward * 2].sigmoid()
        policy = out[:, self.n_reward * 2:self.n_reward * 2 + self.n_act].tanh()
        delta = out[:, -self.n_reward:].tanh()

        return reward, estimation, policy, delta

    # returns shape batch_size * 5
    def loss(self, data):
        obs_in, act_in, obs_next_in, reward_in = data
        batch_size = obs_in.shape[0]
        if self.is_cuda:
            obs_in = obs_in.cuda()
            act_in = act_in.cuda()
            obs_next_in = obs_next_in.cuda()
            reward_in = reward_in.cuda()

        reward, estimation, policy, delta = self((obs_in, act_in))
        _, e_next, _, _ = self((obs_next_in, None))

        loss_reward = functional.binary_cross_entropy(reward, reward_in, reduction='none').mean(dim=1)
        loss_estimation = ModelAgent.estimator_loss(estimation, e_next, reward_in)
        loss_delta = ModelAgent.delta_loss(estimation, e_next, delta)
        loss_policy = self.policy_loss(policy, obs_in, self.reward_weights)

        loss = torch.cat([loss_reward.reshape(batch_size, 1),
                          loss_estimation[0].reshape(batch_size, 1),
                          loss_estimation[1].reshape(batch_size, 1),
                          loss_delta.reshape(batch_size, 1),
                          loss_policy.reshape(batch_size, 1)],
                         dim=1)

        return loss

    @staticmethod
    def estimator_loss(estimation, e_next, reward_in):

        loss_bce = functional.binary_cross_entropy(estimation, reward_in, reduction='none').mean(dim=1)

        # difference to next estimation -> function should be continuous
        loss_difference = ((estimation - e_next) ** 2).mean(dim=1)

        return loss_bce, loss_difference

    @staticmethod
    def delta_loss(estimation, e_next, delta):
        # batch statistics
        delta_real = (e_next - estimation).detach()
        mean = delta_real.mean(dim=0)
        std = ((delta_real - mean) ** 2).mean(dim=0).sqrt()

        delta_real_normal = (delta_real - mean) / std

        return functional.mse_loss(delta, delta_real_normal, reduction='none').mean(dim=1)

    def policy_loss(self, policy, obs_in, reward_weights):
        with torch.enable_grad():
            pc = policy.detach().requires_grad_()

            _, _, _, delta_pol = self((obs_in, pc))

            ((-delta_pol * reward_weights).sum(dim=1)).mean().backward()

            grad = pc.grad
            self.zero_grad()

            loss = .5 * (policy - (pc - grad)) ** 2  # policy grad will be the same as pc grad

            return loss.mean(dim=1)

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
