import torch


class Actor:
    def __init__(self, is_agent):
        self.is_agent = is_agent

    def get_action(self, observation, user_input, nn_step):
        raise NotImplementedError


class ActorNN(Actor):
    def __init__(self, agent, noise=.5, max_speed=1.5):
        super().__init__(True)
        self.agent = agent
        self.noise = noise
        self.max_speed = max_speed
        self.act = torch.FloatTensor().new_zeros((1, agent.n_act))

    def get_action(self, obs, user_input, nn_step):
        if nn_step:
            with torch.no_grad():
                noise_normal = torch.randn(obs.shape[0], self.agent.n_act)
                act = self.agent((obs, None))[2].cpu()

                self.act = act * (1 + self.noise * noise_normal) * self.max_speed

        return self.act


class ActorEps(Actor):
    def __init__(self, agent, epsilon=.5, max_speed=1.5):
        super().__init__(True)
        self.agent = agent
        self.epsilon = epsilon
        self.max_speed = max_speed
        self.act = torch.FloatTensor().new_zeros((1, agent.n_act))

    def get_action(self, obs, user_input, nn_step):
        if nn_step:
            with torch.no_grad():
                act = self.agent((obs, None))[2].cpu()\
                    if torch.rand(1) < self.epsilon else\
                    torch.randn(obs.shape[0], self.agent.n_act)

                self.act = act * self.max_speed

        return self.act


class ActorRandom(Actor):
    def __init__(self, n_act, noise=.5, max_speed=1.5):
        super().__init__(False)
        self.n_act = n_act
        self.noise = noise
        self.max_speed = max_speed
        self.act = torch.FloatTensor().new_zeros((1, n_act))

    def get_action(self, observation, user_input, nn_step):
        if nn_step:
            noise_normal = torch.randn(observation.shape[0], self.n_act)
            act = torch.randn(observation.shape[0], self.n_act)

            self.act = act * (1 + self.noise * noise_normal) * self.max_speed

        return self.act


class ActorDodgeHuman(Actor):
    def __init__(self):
        super().__init__(False)

    def get_action(self, observation, user_input, nn_step):
        x = 1 if user_input.right else (-1 if user_input.left else 0)
        y = 1 if user_input.down else (-1 if user_input.up else 0)
        return torch.FloatTensor().new_tensor([x, y]).reshape((1, 2))
