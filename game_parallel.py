import torch

import dodge_config
import ph_config


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
        super().__init__(n_parallel, dodge_config.OBS_SIZE)

        self.speed_factor_x = dodge_config.OBJECT_SPEED * 2 / dodge_config.WIDTH
        self.speed_factor_y = dodge_config.OBJECT_SPEED * 2 / dodge_config.HEIGHT
        self.present_dist_x = (dodge_config.PLAYERWIDTH + dodge_config.PRESENTWIDTH) * 2 / dodge_config.WIDTH
        self.present_dist_y = (dodge_config.PLAYERWIDTH + dodge_config.PRESENTHEIGHT) * 2 / dodge_config.HEIGHT
        self.obst_dist_x = (dodge_config.OBSTACLEWIDTH + dodge_config.PLAYERWIDTH) * 2 / dodge_config.WIDTH
        self.obst_dist_y = (dodge_config.OBSTACLEWIDTH + dodge_config.PLAYERWIDTH) * 2 / dodge_config.HEIGHT

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
        self.obs[:, dodge_config.SPLIT[2]:dodge_config.SPLIT[4]] = \
            torch.where(self.obs[:, 2:3] > 1,
                        torch.rand(self.obs_size[0], dodge_config.OBJECTSHOR * 2) * 2 - 1,
                        self.obs[:, dodge_config.SPLIT[2]:dodge_config.SPLIT[4]])

        # shuffle obstacles up/down
        self.obs[:, dodge_config.SPLIT[4]:dodge_config.SPLIT[6]] = \
            torch.where(self.obs[:, 3:4] > 1,
                        torch.rand(self.obs_size[0], dodge_config.OBJECTSVER * 2) * 2 - 1,
                        self.obs[:, dodge_config.SPLIT[4]:dodge_config.SPLIT[6]])

        # reset obstacle distance
        self.obs[:, 2:4] = torch.where(self.obs[:, 2:4] > 1, torch.FloatTensor().new_tensor(-1), self.obs[:, 2:4])

        # reset present
        self.obs[:, dodge_config.SPLIT[6]:] = torch.where(
            torch.cat([
                      ((self.obs[:, 0:2] - self.obs[:, dodge_config.SPLIT[6]:dodge_config.OBS_SIZE]).abs() <
                           torch.FloatTensor().new_tensor([self.present_dist_x / 2, self.present_dist_y / 2]))
                      .all(dim=1).reshape(self.obs_size[0], 1)
                      ] * 2, dim=1),
            torch.rand(self.obs_size[0], 2) * 2 - 1,
            self.obs[:, dodge_config.SPLIT[6]:]
        )

    def rewards(self):
        present = ((self.obs[:, 0:2] - self.obs[:, dodge_config.SPLIT[6]:]).abs() < torch.FloatTensor().new_tensor(
            [self.present_dist_x / 2, self.present_dist_y / 2])).all(dim=1).reshape(self.obs_size[0], 1)

        right = ((self.obs[:, 0:1] - self.obs[:, 2:3]).abs() < self.obst_dist_x / 2).flatten() & \
                ((self.obs[:, 1:2] - self.obs[:, dodge_config.SPLIT[2]:dodge_config.SPLIT[3]]).abs() <
                 self.obst_dist_y / 2).any(dim=1)

        left = ((self.obs[:, 0:1] + self.obs[:, 2:3]).abs() < self.obst_dist_x / 2).flatten() & \
               ((self.obs[:, 1:2] - self.obs[:, dodge_config.SPLIT[3]:dodge_config.SPLIT[4]]).abs() <
                self.obst_dist_y / 2).any(dim=1)

        down = ((self.obs[:, 0:1] - self.obs[:, dodge_config.SPLIT[4]:dodge_config.SPLIT[5]]).abs() <
                self.obst_dist_x / 2).any(dim=1) & \
               ((self.obs[:, 1:2] - self.obs[:, 3:4]).abs() < self.obst_dist_y / 2).flatten()

        up = ((self.obs[:, 0:1] - self.obs[:, dodge_config.SPLIT[5]:dodge_config.SPLIT[6]]).abs() <
              self.obst_dist_x / 2).any(dim=1) & \
             ((self.obs[:, 1:2] + self.obs[:, 3:4]).abs() < self.obst_dist_y / 2).flatten()

        obstacle = (right | left | down | up).reshape(self.obs_size[0], 1)

        return torch.cat([obstacle, present], dim=1)


class PresentHunterParallel(GameParallel):
    def __init__(self, n_parallel=1):
        super().__init__(n_parallel, 4)

        self.speed_factor_x = ph_config.PLAYERSPEED * 2 / ph_config.WIDTH
        self.speed_factor_y = ph_config.PLAYERSPEED * 2 / ph_config.HEIGHT
        self.present_dist_x = (ph_config.PLAYERWIDTH + ph_config.PRESENTWIDTH) * 2 / ph_config.WIDTH
        self.present_dist_y = (ph_config.PLAYERWIDTH + ph_config.PRESENTHEIGHT) * 2 / ph_config.HEIGHT

    def initial_obs(self):
        obs = torch.rand(self.obs_size) * 2 - 1
        obs[:, :2] = obs[:, :2] * .5  # player position is in center

        return obs

    def action_move(self, action, time_elapsed, speed):
        self.obs[:, 0:1] = self.obs[:, 0:1] + action[:, 0:1] * self.speed_factor_x * time_elapsed * speed
        self.obs[:, 1:2] = self.obs[:, 1:2] + action[:, 1:2] * self.speed_factor_y * time_elapsed * speed

    def action_triggered(self):
        # align player
        self.obs[:, 0:2] = torch.clamp(self.obs[:, 0:2], -1, 1)

        # reset present
        self.obs[:, 2:4] = torch.where(
            torch.cat([
                          ((self.obs[:, 0:2] - self.obs[:, 2:4]).abs() <
                           torch.FloatTensor().new_tensor([self.present_dist_x / 2, self.present_dist_y / 2]))
                      .all(dim=1).reshape(self.obs_size[0], 1)
                      ] * 2, dim=1),
            torch.rand(self.obs_size[0], 2) * 2 - 1,
            self.obs[:, 2:4]
        )

    def rewards(self):
        present = ((self.obs[:, 0:2] - self.obs[:, 2:4]).abs() < torch.FloatTensor().new_tensor(
            [self.present_dist_x / 2, self.present_dist_y / 2])).all(dim=1).reshape(self.obs_size[0], 1)

        return present
