import pygame
import torch
import numpy
import math

import util


class GameUI:
    def __init__(self):
        pass

    def draw(self, window, actor, font, obs):
        raise NotImplementedError

    def draw_pause(self, window, actor, font, obs):
        raise NotImplementedError

    def update_pause(self, actor, obs, user_input):
        raise NotImplementedError


class DodgeUI(GameUI):
    def __init__(self):
        super().__init__()

        self.backgroundTimer = 0

        # image creation
        self.img_player = pygame.image.load("resources/character.png")
        self.img_obstacle = pygame.image.load("resources/obstacle.png")
        self.img_present = pygame.image.load("resources/present.png")

        self.player_trans = pygame.Surface((util.PLAYERWIDTH, util.PLAYERWIDTH))
        self.player_trans.set_alpha(100)
        self.player_trans.fill((0, 0, 0))

        self.obs_trans = pygame.Surface((util.OBSTACLEWIDTH, util.OBSTACLEWIDTH))
        self.obs_trans.set_alpha(100)
        self.obs_trans.fill((200, 0, 0))

        self.present_trans = pygame.Surface((util.PRESENTWIDTH, util.PRESENTHEIGHT))
        self.present_trans.set_alpha(100)
        self.present_trans.fill((0, 150, 0))

        # mouse selection
        self.selected = -1
        self.override_action = None

    def draw(self, window, actor, font, obs):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(obs)

        window.fill((int(255 * 0.5 * (1 + math.sin(0.001 * self.backgroundTimer))),
                     int(255 * 0.5 * (1 + math.sin(0.001 * self.backgroundTimer + 2 * math.pi / 3))),
                     int(255 * 0.5 * (1 + math.sin(0.001 * self.backgroundTimer + 4 * math.pi / 3)))))

        window.blit(self.img_present, (presentx - util.PRESENTWIDTH / 2, presenty - util.PRESENTHEIGHT / 2))

        for i in range(5):
            window.blit(self.img_obstacle, (disth - util.OBSTACLEWIDTH / 2, righty[i] - util.OBSTACLEWIDTH / 2))
        for i in range(5):
            window.blit(self.img_obstacle,
                        (util.WIDTH - disth - util.OBSTACLEWIDTH / 2, lefty[i] - util.OBSTACLEWIDTH / 2))
        for i in range(7):
            window.blit(self.img_obstacle, (downx[i] - util.OBSTACLEWIDTH / 2, distv - util.OBSTACLEWIDTH / 2))
        for i in range(7):
            window.blit(self.img_obstacle,
                        (upx[i] - util.OBSTACLEWIDTH / 2, util.HEIGHT - distv - util.OBSTACLEWIDTH / 2))
        window.blit(self.img_player, (playerx - util.PLAYERWIDTH / 2, playery - util.PLAYERWIDTH / 2))

        if actor.is_agent:
            with torch.no_grad():
                p = actor.agent.get_reward(obs).cpu()
            for i in range(p.shape[1]):
                text_surface = font.render('Reward: {:.4f}'.format(p.numpy()[0, i]), False, (0, 0, 0))
                window.blit(text_surface, (0, 50 * i))

    def update_pause(self, actor, obs, user_input):
        left, right, wheel = pygame.mouse.get_pressed()
        if user_input.mouse_left:
            self.selected = self.get_picked(user_input.mouse_x, user_input.mouse_y, obs)
            self.override_action = None
        if self.selected >= 0:
            if left:
                obs = self.move_observation(user_input.mouse_x, user_input.mouse_y, obs)
            else:
                self.selected = -1
        else:
            # if no object is clicked change player action
            if left:
                playerx, playery, _, _, _, _, _, _, _, _ = self.decode_obs(obs)
                act_x = (user_input.mouse_x - playerx) / (util.PLAYERSPEED * util.ACT_DRAW_LENGTH)
                act_y = (user_input.mouse_y - playery) / (util.PLAYERSPEED * util.ACT_DRAW_LENGTH)

                self.override_action = torch.FloatTensor().new_tensor([act_x, act_y]).reshape(1, 2)
        return obs

    def draw_pause(self, window, actor, font, obs):
        if actor.is_agent:
            playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(obs)

            act = torch.FloatTensor().new_zeros(1, 2)
            grad = actor.agent.reward_gradient(obs, torch.FloatTensor().new_tensor([-1, .5])).cpu()
            grad_fac = .1

            obs_shadow = obs + grad * grad_fac
            f_playerx, f_playery, f_disth, f_distv, f_righty, f_lefty, f_downx, f_upx, f_presentx, f_presenty = \
                self.decode_obs(obs_shadow)

            window.blit(self.present_trans, (f_presentx - util.PRESENTWIDTH / 2, f_presenty - util.PRESENTWIDTH / 2))
            for i in range(5):
                window.blit(self.obs_trans, (f_disth - util.OBSTACLEWIDTH / 2, f_righty[i] - util.OBSTACLEWIDTH / 2))
            for i in range(5):
                window.blit(self.obs_trans,
                            (util.WIDTH - f_disth - util.OBSTACLEWIDTH / 2, f_lefty[i] - util.OBSTACLEWIDTH / 2))
            for i in range(7):
                window.blit(self.obs_trans, (f_downx[i] - util.OBSTACLEWIDTH / 2, f_distv - util.OBSTACLEWIDTH / 2))
            for i in range(7):
                window.blit(self.obs_trans,
                            (f_upx[i] - util.OBSTACLEWIDTH / 2, util.HEIGHT - f_distv - util.OBSTACLEWIDTH / 2))
            window.blit(self.player_trans, (f_playerx - util.PLAYERWIDTH / 2, f_playery - util.PLAYERWIDTH / 2))

    # 0: player, 12345: r, 678910: l 1234567: d, 8901234: u, 25: present
    def get_picked(self, x, y, obs):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(obs)
        if util.is_in_rect(x, y, playerx, playery, util.PLAYERWIDTH, util.PLAYERWIDTH):
            return 0
        for i in range(5):
            if util.is_in_rect(x, y, disth, righty[i], util.OBSTACLEWIDTH, util.OBSTACLEWIDTH):
                return i + 1
        for i in range(5):
            if util.is_in_rect(x, y, util.WIDTH - disth, lefty[i], util.OBSTACLEWIDTH, util.OBSTACLEWIDTH):
                return i + 6
        for i in range(7):
            if util.is_in_rect(x, y, downx[i], distv, util.OBSTACLEWIDTH, util.OBSTACLEWIDTH):
                return i + 11
        for i in range(7):
            if util.is_in_rect(x, y, upx[i], util.HEIGHT - distv, util.OBSTACLEWIDTH, util.OBSTACLEWIDTH):
                return i + 18
        if util.is_in_rect(x, y, presentx, presenty, util.PRESENTWIDTH, util.PRESENTHEIGHT):
            return 25
        return -1

    def move_observation(self, mouse_x, mouse_y, obs):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(obs)

        if self.selected < 0:
            return
        if self.selected < 1:
            playerx = mouse_x
            playery = mouse_y
        elif self.selected < 6:
            disth = mouse_x
            righty[self.selected - 1] = mouse_y
        elif self.selected < 11:
            disth = util.WIDTH - mouse_x
            lefty[self.selected - 6] = mouse_y
        elif self.selected < 18:
            downx[self.selected - 11] = mouse_x
            distv = mouse_y
        elif self.selected < 25:
            upx[self.selected - 18] = mouse_x
            distv = util.HEIGHT - mouse_y
        elif self.selected == 25:
            presentx = mouse_x
            presenty = mouse_y

        return self.encode_obs((playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty))

    @staticmethod
    def decode_obs(obs):
        playerx = util.toDrawScaleX(obs[0, 0].numpy())
        playery = util.toDrawScaleY(obs[0, 1].numpy())
        disth = util.toDrawScaleX(obs[0, 2].numpy())
        distv = util.toDrawScaleY(obs[0, 3].numpy())
        righty = util.toDrawScaleY(obs[0, 4:9].numpy())
        lefty = util.toDrawScaleY(obs[0, 9:14].numpy())
        downx = util.toDrawScaleX(obs[0, 14:21].numpy())
        upx = util.toDrawScaleX(obs[0, 21:28].numpy())
        presentx = util.toDrawScaleX(obs[0, 28].numpy())
        presenty = util.toDrawScaleY(obs[0, 29].numpy())

        return playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty

    @staticmethod
    def encode_obs(values):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = values

        return torch.FloatTensor().new_tensor(
            numpy.concatenate((util.toEncodeScaleX(numpy.array([playerx])).reshape(1),
                               util.toEncodeScaleY(numpy.array([playery])).reshape(1),
                               util.toEncodeScaleX(numpy.array([disth])).reshape(1),
                               util.toEncodeScaleY(numpy.array([distv])).reshape(1),
                               util.toEncodeScaleY(numpy.array([righty])).reshape(5),
                               util.toEncodeScaleY(numpy.array([lefty])).reshape(5),
                               util.toEncodeScaleX(numpy.array([downx])).reshape(7),
                               util.toEncodeScaleX(numpy.array([upx])).reshape(7),
                               util.toEncodeScaleX(numpy.array([presentx])).reshape(1),
                               util.toEncodeScaleY(numpy.array([presenty])).reshape(1)
                               ))).reshape((1, 30))
