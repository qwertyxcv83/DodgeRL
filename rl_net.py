import torch
import torch.nn.functional as functional
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import FloatTensor
import nn_util


class RewardModel(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(RewardModel, self).__init__()

        self.layer = nn_util.MaxDenseResNet(n_in, depth=4, n_max=2)
        self.out = torch.nn.Linear(n_in, n_out)

    def forward(self, obs):

        x = self.layer(obs)

        return self.out(x).sigmoid()


class PolicyModel(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(PolicyModel, self).__init__()
        self.lin = nn_util.WeightedAverageLayer(n_in, n_out, 100)

    def forward(self, obs):

        out = self.lin(obs)

        return out


class NextHiddenModel(torch.nn.Module):
    def __init__(self, n_in, n_out, n_soft=10):
        super(NextHiddenModel, self).__init__()

        self.layer = nn_util.WeightedAverageLayer(n_in, n_out, n_soft)

    def forward(self, values):
        hidden, act = values
        hidden_flat = hidden.reshape(hidden.shape[0], hidden.shape[1:].numel())

        hid_act = torch.cat([hidden_flat, act], 1)

        next_hidden = hidden + self.layer(hid_act).reshape(hidden.shape)

        return next_hidden


class ModelAgent(torch.nn.Module):
    def __init__(self, n_obs, n_act, n_rewards):
        super(ModelAgent, self).__init__()

        self.n_obs = n_obs
        self.n_act = n_act
        self.n_rewards = n_rewards

        self.next_hidden = NextHiddenModel(n_obs + n_act, n_obs)
        self.reward = RewardModel(n_obs, n_rewards)
        self.policy = PolicyModel(n_obs, n_act)

    def forward(self, values):
        raise NotImplementedError

    def get_action(self, obs):
        rew_grad = self.reward_gradient(obs, torch.FloatTensor().new_tensor([-1, .5]))
        return rew_grad[:, :2].tanh()
        # return torch.randn(2)
        # return torch.tanh(self.policy(self.hidden(obs)))

    def get_reward(self, obs):
        return self.reward(obs)

    def reward_gradient(self, obs, rewards):
        obs.requires_grad_()
        if obs.grad is not None:
            obs.grad.zero_()

        rew = self.reward(obs) * rewards.reshape(1, 2)
        rew.sum().backward()

        grad = obs.grad
        obs.requires_grad_(False)
        return grad

    def action_gradient(self, values, rewards):
        obs, act = values

        act.requires_grad_()
        if act.grad is not None:
            act.grad.detach_()
            act.grad.zero_()
        self.zero_grad()

        next_hidden = self.next_hidden((obs, act))
        E = self.reward(next_hidden)

        minim = - (rewards * E).mean()

        minim.backward()

        grad = act.grad
        act.requires_grad_(False)
        return grad

    def loss(self, obs_in, act_in, obs_next_in, reward_time_in, reward_bool_in):
        reward = self.reward(obs_in)

        loss_rew = functional.binary_cross_entropy(reward, reward_bool_in)

        return loss_rew, FloatTensor().new_tensor([0]), FloatTensor().new_tensor([0]), FloatTensor().new_tensor([0]), FloatTensor().new_tensor([0])


def train(epochs, model_agent, train_set, test_set, rewards, lr_rew=.01, lr_next=.01, lr_pol=.01,
          batch_size_train=128, batch_size_test=128, c1=1, c2=2, c3=.5, plot=True, des_loss=float('inf'), zero_step=False):
    n_losses = 5

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False, drop_last=True)

    train_epoch_losses = torch.FloatTensor().new_zeros((0, n_losses))
    test_epoch_losses = torch.FloatTensor().new_zeros((0, n_losses))

    train_losses = torch.FloatTensor().new_zeros((0, n_losses))
    test_losses = torch.FloatTensor().new_zeros((0, n_losses))

    optimizer_reward = torch.optim.SGD(model_agent.reward.parameters(), lr=lr_rew)
    optimizer_next_hidden = torch.optim.SGD(model_agent.next_hidden.parameters(), lr=lr_next)
    optimizer_policy = torch.optim.SGD(model_agent.policy.parameters(), lr=lr_pol)
    model_agent.train()

    for epoch in range(epochs):
        print("training epoch {}".format(epoch), end="")
        steps = 0
        batches = 0
        for obs_in, act_in, obs_next_in, reward_time_in, reward_bool_in in train_loader:
            batches += 1

            loss_reward, loss_next_hidden, loss_policy_rew, loss_policy_center, loss_policy_entropy = \
                model_agent.loss(obs_in, act_in, obs_next_in, reward_time_in, reward_bool_in)

            if not zero_step or loss_reward > des_loss:
                model_agent.zero_grad()
                loss_reward.backward()
                optimizer_reward.step()
                steps += 1

            losses = torch.cat([loss_reward.detach().reshape(1),
                                loss_next_hidden.detach().reshape(1),
                                loss_policy_rew.detach().reshape(1),
                                loss_policy_center.detach().reshape(1),
                                loss_policy_entropy.detach().reshape(1)])
            train_losses = torch.cat([train_losses, losses.reshape(1, n_losses)], 0)

            while loss_reward > des_loss:
                loss_reward, _, _, _, _ = model_agent.loss(obs_in, act_in, obs_next_in, reward_time_in, reward_bool_in)
                model_agent.zero_grad()
                loss_reward.backward()
                optimizer_reward.step()
                steps += 1

        print(", steps: {:.4f}".format(float(steps) / batches), end="")

        with torch.no_grad():
            model_agent.eval()

            for obs_in, act_in, obs_next_in, reward_time_in, reward_bool_in in test_loader:

                loss_reward, loss_next_hidden, loss_policy_rew, loss_policy_center, loss_policy_entropy = \
                    model_agent.loss(obs_in, act_in, obs_next_in, reward_time_in, reward_bool_in)

                losses = torch.cat([loss_reward.reshape(1),
                                    loss_next_hidden.reshape(1),
                                    loss_policy_rew.reshape(1),
                                    loss_policy_center.reshape(1),
                                    loss_policy_entropy.reshape(1)])
                test_losses = torch.cat([test_losses, losses.reshape(1, n_losses)])

            model_agent.train()

        train_losses_mean = train_losses.mean(dim=0).reshape(1, n_losses)
        test_losses_mean = test_losses.mean(dim=0).reshape(1, n_losses)

        train_epoch_losses = torch.cat([train_epoch_losses, train_losses_mean])
        test_epoch_losses = torch.cat([test_epoch_losses, test_losses_mean])

        print(", mean training losses: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}"
              ", mean test loss: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}"
              .format(train_losses_mean[0, 0],
                      train_losses_mean[0, 1],
                      train_losses_mean[0, 2],
                      train_losses_mean[0, 3],
                      train_losses_mean[0, 4],
                      test_losses_mean[0, 0],
                      test_losses_mean[0, 1],
                      test_losses_mean[0, 2],
                      test_losses_mean[0, 3],
                      test_losses_mean[0, 4]))

        train_losses = torch.FloatTensor().new_zeros((0, n_losses))
        test_losses = torch.FloatTensor().new_zeros((0, n_losses))

    model_agent.eval()

    if plot:
        plt.scatter(range(epochs), train_epoch_losses[:, 0], color='r', s=10, marker='o')
        plt.scatter(range(epochs), test_epoch_losses[:, 0], color='g', s=10, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Reward Loss")
        plt.figure()
        plt.scatter(range(epochs), train_epoch_losses[:, 1], color='r', s=10, marker='o')
        plt.scatter(range(epochs), test_epoch_losses[:, 1], color='g', s=10, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("World Loss")
        plt.figure()
        plt.scatter(range(epochs), train_epoch_losses[:, 2], color='r', s=10, marker='o')
        plt.scatter(range(epochs), test_epoch_losses[:, 2], color='g', s=10, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Policy Reward Loss")
        plt.figure()
        plt.scatter(range(epochs), train_epoch_losses[:, 3], color='r', s=10, marker='o')
        plt.scatter(range(epochs), test_epoch_losses[:, 3], color='g', s=10, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Policy Center Loss")
        plt.figure()
        plt.scatter(range(epochs), train_epoch_losses[:, 4], color='r', s=10, marker='o')
        plt.scatter(range(epochs), test_epoch_losses[:, 4], color='g', s=10, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Policy Entropy Loss")

        plt.show()

    return train_epoch_losses, test_epoch_losses
