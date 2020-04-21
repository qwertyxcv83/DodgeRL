import torch
import util


class GameParallel:
    def __init__(self, n_parallel, n_obs):
        self.obs_size = torch.Size([n_parallel, n_obs])
        self.obs = None

    def initial_obs(self):
        raise NotImplementedError

    def action_move(self, action, time_elapsed, speed):
        raise NotImplementedError

    def action_triggered(self):
        raise NotImplementedError

    def rewards(self):
        raise NotImplementedError


class DodgeParallel(GameParallel):
    def __init__(self, n_parallel=1):
        super().__init__(n_parallel, 30)

        self.speed_factor_x = util.PLAYERSPEED * 2 / util.WIDTH
        self.speed_factor_y = util.PLAYERSPEED * 2 / util.HEIGHT
        self.present_dist_x = (util.PLAYERWIDTH + util.PRESENTWIDTH) * 2 / util.WIDTH
        self.present_dist_y = (util.PLAYERWIDTH + util.PRESENTHEIGHT) * 2 / util.HEIGHT
        self.obst_dist_x = (util.OBSTACLEWIDTH + util.PLAYERWIDTH) * 2 / util.WIDTH
        self.obst_dist_y = (util.OBSTACLEWIDTH + util.PLAYERWIDTH) * 2 / util.HEIGHT

    def initial_obs(self):
        obs = torch.rand(self.obs_size) * 2 - 1
        obs[:, :2] = obs[:, :2] * .5  # player position is in center
        obs[:, 2:4] = -1  # set obstacles at the wall

        return obs

    def action_move(self, action, time_elapsed, speed):
        self.obs[:, 0:1] = self.obs[:, 0:1] + action[:, 0:1] * self.speed_factor_x * time_elapsed * speed
        self.obs[:, 1:2] = self.obs[:, 1:2] + action[:, 1:2] * self.speed_factor_y * time_elapsed * speed
        self.obs[:, 2:3] = self.obs[:, 2:3] + self.speed_factor_x * time_elapsed * speed
        self.obs[:, 3:4] = self.obs[:, 3:4] + self.speed_factor_y * time_elapsed * speed

    def action_triggered(self):
        # align player
        self.obs[:, 0:2] = torch.clamp(self.obs[:, 0:2], -1, 1)

        # shuffle obstacles left/right
        self.obs[:, 4:14] = torch.where(self.obs[:, 2:3] > 1, torch.rand(self.obs_size[0], 10), self.obs[:, 4:14])
        # shuffle obstacles up/down
        self.obs[:, 14:28] = torch.where(self.obs[:, 3:4] > 1, torch.rand(self.obs_size[0], 14), self.obs[:, 14:28])
        # reset obstacle distance
        self.obs[:, 2:4] = torch.where(self.obs[:, 2:4] > 1, torch.FloatTensor().new_tensor(-1), self.obs[:, 2:4])

        # reset present
        self.obs[:, 28:30] = torch.where(
            torch.cat([
                          ((self.obs[:, 0:2] - self.obs[:, 28:30]).abs() <
                           torch.FloatTensor().new_tensor([self.present_dist_x / 2, self.present_dist_y / 2]))
                      .all(dim=1).reshape(self.obs_size[0], 1)
                      ] * 2, dim=1),
            torch.rand(self.obs_size[0], 2),
            self.obs[:, 28:30]
        )

    def rewards(self):
        present = ((self.obs[:, 0:2] - self.obs[:, 28:30]).abs() < torch.FloatTensor().new_tensor(
            [self.present_dist_x / 2, self.present_dist_y / 2])).all(dim=1).reshape(self.obs_size[0], 1)

        right = ((self.obs[:, 0:1] - self.obs[:, 2:3]).abs() < self.obst_dist_x / 2).flatten() & \
                ((self.obs[:, 1:2] - self.obs[:, 4:9]).abs() < self.obst_dist_y / 2).any(dim=1)

        left = ((self.obs[:, 0:1] + self.obs[:, 2:3]).abs() < self.obst_dist_x / 2).flatten() & \
               ((self.obs[:, 1:2] - self.obs[:, 9:14]).abs() < self.obst_dist_y / 2).any(dim=1)

        down = ((self.obs[:, 0:1] - self.obs[:, 14:21]).abs() < self.obst_dist_x / 2).any(dim=1) & \
               ((self.obs[:, 1:2] - self.obs[:, 3:4]).abs() < self.obst_dist_y / 2).flatten()

        up = ((self.obs[:, 0:1] - self.obs[:, 21:28]).abs() < self.obst_dist_x / 2).any(dim=1) & \
             ((self.obs[:, 1:2] - self.obs[:, 3:4]).abs() < self.obst_dist_y / 2).flatten()

        obstacle = (right | left | down | up).reshape(self.obs_size[0], 1)

        return torch.cat([obstacle, present], dim=1)
