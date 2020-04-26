import pygame
import torch
import numpy
import math

import dodge_config
import ph_config
import util


class GameUI:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def draw(self, window, actor, font, game):
        raise NotImplementedError

    def draw_pause(self, window, actor, font, game):
        raise NotImplementedError

    def update_pause(self, actor, obs, user_input):
        raise NotImplementedError


class DodgeUI(GameUI):
    def __init__(self):
        super().__init__(dodge_config.WIDTH, dodge_config.HEIGHT)

        self.backgroundTimer = 0

        # image creation
        self.img_player = pygame.image.load("resources/character.png")
        self.img_obstacle = pygame.image.load("resources/obstacle.png")
        self.img_present = pygame.image.load("resources/present.png")

        self.player_trans = pygame.Surface((dodge_config.PLAYERWIDTH, dodge_config.PLAYERWIDTH))
        self.player_trans.set_alpha(100)
        self.player_trans.fill((0, 0, 0))

        self.obs_trans = pygame.Surface((dodge_config.OBSTACLEWIDTH, dodge_config.OBSTACLEWIDTH))
        self.obs_trans.set_alpha(100)
        self.obs_trans.fill((200, 0, 0))

        self.present_trans = pygame.Surface((dodge_config.PRESENTWIDTH, dodge_config.PRESENTHEIGHT))
        self.present_trans.set_alpha(100)
        self.present_trans.fill((0, 150, 0))

        # mouse selection
        self.selected = -1
        self.override_action = None

    def draw(self, window, actor, font, game):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(game.obs)
        reward = game.rewards().any()

        if reward:
            window.fill((255, 255, 255))
        else:
            window.fill((int(255 * 0.5 * (1 + math.sin(0.001 * self.backgroundTimer))),
                        int(255 * 0.5 * (1 + math.sin(0.001 * self.backgroundTimer + 2 * math.pi / 3))),
                        int(255 * 0.5 * (1 + math.sin(0.001 * self.backgroundTimer + 4 * math.pi / 3)))))

        window.blit(self.img_present, (presentx - dodge_config.PRESENTWIDTH / 2, presenty - dodge_config.PRESENTHEIGHT / 2))

        for i in range(dodge_config.OBJECTSHOR):
            window.blit(self.img_obstacle, (disth - dodge_config.OBSTACLEWIDTH / 2, righty[i] - dodge_config.OBSTACLEWIDTH / 2))
        for i in range(dodge_config.OBJECTSHOR):
            window.blit(self.img_obstacle,
                        (dodge_config.WIDTH - disth - dodge_config.OBSTACLEWIDTH / 2, lefty[i] - dodge_config.OBSTACLEWIDTH / 2))
        for i in range(dodge_config.OBJECTSVER):
            window.blit(self.img_obstacle, (downx[i] - dodge_config.OBSTACLEWIDTH / 2, distv - dodge_config.OBSTACLEWIDTH / 2))
        for i in range(dodge_config.OBJECTSVER):
            window.blit(self.img_obstacle,
                        (upx[i] - dodge_config.OBSTACLEWIDTH / 2, dodge_config.HEIGHT - distv - dodge_config.OBSTACLEWIDTH / 2))
        window.blit(self.img_player, (playerx - dodge_config.PLAYERWIDTH / 2, playery - dodge_config.PLAYERWIDTH / 2))

        if actor.is_agent:
            with torch.no_grad():
                r, e, _, _ = actor.agent((game.obs, None))
                r = r.cpu()
                e = e.cpu()
            for i in range(r.shape[1]):
                text_surface = font.render('Rew: {:.4f}, Est: {:.4f}'.format(r.numpy()[0, i], e.numpy()[0, i]), False, (0, 0, 0))
                window.blit(text_surface, (0, 50 * i))

    def update_pause(self, actor, game, user_input):
        left, right, wheel = pygame.mouse.get_pressed()
        if user_input.mouse_left:
            self.selected = self.get_picked(user_input.mouse_x, user_input.mouse_y, game.obs)
            self.override_action = None
        if self.selected >= 0:
            if left:
                game.obs = self.move_observation(user_input.mouse_x, user_input.mouse_y, game.obs)
            else:
                self.selected = -1
        else:
            # if no object is clicked change player action
            if left:
                playerx, playery, _, _, _, _, _, _, _, _ = self.decode_obs(game.obs)
                act_x = (user_input.mouse_x - playerx) / (dodge_config.OBJECT_SPEED * dodge_config.ACT_DRAW_LENGTH)
                act_y = (user_input.mouse_y - playery) / (dodge_config.OBJECT_SPEED * dodge_config.ACT_DRAW_LENGTH)

                self.override_action = torch.FloatTensor().new_tensor([act_x, act_y]).reshape(1, 2)

    def draw_pause(self, window, actor, font, game):
        if actor.is_agent:
            grad = actor.agent.reward_gradient(game.obs, torch.FloatTensor().new_tensor([-1, .5])).cpu()
            grad_fac = .1

            obs_shadow = game.obs + grad * grad_fac
            f_playerx, f_playery, f_disth, f_distv, f_righty, f_lefty, f_downx, f_upx, f_presentx, f_presenty = \
                self.decode_obs(obs_shadow)

            window.blit(self.present_trans, (f_presentx - dodge_config.PRESENTWIDTH / 2, f_presenty - dodge_config.PRESENTHEIGHT / 2))
            for i in range(dodge_config.OBJECTSHOR):
                window.blit(self.obs_trans, (f_disth - dodge_config.OBSTACLEWIDTH / 2, f_righty[i] - dodge_config.OBSTACLEWIDTH / 2))
            for i in range(dodge_config.OBJECTSHOR):
                window.blit(self.obs_trans,
                            (dodge_config.WIDTH - f_disth - dodge_config.OBSTACLEWIDTH / 2, f_lefty[i] - dodge_config.OBSTACLEWIDTH / 2))
            for i in range(dodge_config.OBJECTSVER):
                window.blit(self.obs_trans, (f_downx[i] - dodge_config.OBSTACLEWIDTH / 2, f_distv - dodge_config.OBSTACLEWIDTH / 2))
            for i in range(dodge_config.OBJECTSVER):
                window.blit(self.obs_trans,
                            (f_upx[i] - dodge_config.OBSTACLEWIDTH / 2, dodge_config.HEIGHT - f_distv - dodge_config.OBSTACLEWIDTH / 2))
            window.blit(self.player_trans, (f_playerx - dodge_config.PLAYERWIDTH / 2, f_playery - dodge_config.PLAYERWIDTH / 2))

    # 0: player, 12345: r, 678910: l 1234567: d, 8901234: u, 25: present
    def get_picked(self, x, y, obs):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(obs)
        if util.is_in_rect(x, y, playerx, playery, dodge_config.PLAYERWIDTH, dodge_config.PLAYERWIDTH):
            return 0
        for i in range(dodge_config.OBJECTSHOR):
            if util.is_in_rect(x, y, disth, righty[i], dodge_config.OBSTACLEWIDTH, dodge_config.OBSTACLEWIDTH):
                return i + 1
        for i in range(dodge_config.OBJECTSHOR):
            if util.is_in_rect(x, y, dodge_config.WIDTH - disth, lefty[i], dodge_config.OBSTACLEWIDTH, dodge_config.OBSTACLEWIDTH):
                return i + 1 + dodge_config.OBJECTSHOR
        for i in range(dodge_config.OBJECTSVER):
            if util.is_in_rect(x, y, downx[i], distv, dodge_config.OBSTACLEWIDTH, dodge_config.OBSTACLEWIDTH):
                return i + 1 + dodge_config.OBJECTSHOR * 2
        for i in range(dodge_config.OBJECTSVER):
            if util.is_in_rect(x, y, upx[i], dodge_config.HEIGHT - distv, dodge_config.OBSTACLEWIDTH, dodge_config.OBSTACLEWIDTH):
                return i + 1 + dodge_config.OBJECTSHOR * 2 + dodge_config.OBJECTSVER
        if util.is_in_rect(x, y, presentx, presenty, dodge_config.PRESENTWIDTH, dodge_config.PRESENTHEIGHT):
            return 1 + dodge_config.OBJECTSHOR * 2 + dodge_config.OBJECTSVER * 2
        return -1

    def move_observation(self, mouse_x, mouse_y, obs):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(obs)

        if self.selected < 0:
            return
        if self.selected < 1:
            playerx = mouse_x
            playery = mouse_y
        elif self.selected < 1 + dodge_config.OBJECTSHOR:
            disth = mouse_x
            righty[self.selected - 1] = mouse_y
        elif self.selected < 1 + dodge_config.OBJECTSHOR * 2:
            disth = dodge_config.WIDTH - mouse_x
            lefty[self.selected - (1 + dodge_config.OBJECTSHOR)] = mouse_y
        elif self.selected < 1 + dodge_config.OBJECTSHOR * 2 + dodge_config.OBJECTSVER:
            downx[self.selected - (1 + dodge_config.OBJECTSHOR * 2)] = mouse_x
            distv = mouse_y
        elif self.selected < 1 + dodge_config.OBJECTSHOR * 2 + dodge_config.OBJECTSVER * 2:
            upx[self.selected - (1 + dodge_config.OBJECTSHOR * 2 + dodge_config.OBJECTSVER)] = mouse_x
            distv = dodge_config.HEIGHT - mouse_y
        elif self.selected == 1 + dodge_config.OBJECTSHOR * 2 + dodge_config.OBJECTSVER * 2:
            presentx = mouse_x
            presenty = mouse_y

        return self.encode_obs((playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty))

    @staticmethod
    def decode_obs(obs):
        playerx = (obs[0, 0].numpy() + 1) / 2 * dodge_config.WIDTH
        playery = (obs[0, 1].numpy() + 1) / 2 * dodge_config.HEIGHT
        disth = (obs[0, 2].numpy() + 1) / 2 * dodge_config.WIDTH
        distv = (obs[0, 3].numpy() + 1) / 2 * dodge_config.HEIGHT
        righty = (obs[0, dodge_config.SPLIT[2]:dodge_config.SPLIT[3]].numpy() + 1) / 2 * dodge_config.HEIGHT
        lefty = (obs[0, dodge_config.SPLIT[3]:dodge_config.SPLIT[4]].numpy() + 1) / 2 * dodge_config.HEIGHT
        downx = (obs[0, dodge_config.SPLIT[4]:dodge_config.SPLIT[5]].numpy() + 1) / 2 * dodge_config.WIDTH
        upx = (obs[0, dodge_config.SPLIT[5]:dodge_config.SPLIT[6]].numpy() + 1) / 2 * dodge_config.WIDTH
        presentx = (obs[0, dodge_config.SPLIT[6]].numpy() + 1) / 2 * dodge_config.WIDTH
        presenty = (obs[0, dodge_config.SPLIT[6] + 1].numpy() + 1) / 2 * dodge_config.HEIGHT

        return playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty

    @staticmethod
    def encode_obs(values):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = values

        return torch.FloatTensor().new_tensor(
            numpy.concatenate(((numpy.array([playerx]) / dodge_config.WIDTH * 2 - 1).reshape(1),
                               (numpy.array([playery]) / dodge_config.HEIGHT * 2 - 1).reshape(1),
                               (numpy.array([disth]) / dodge_config.WIDTH * 2 - 1).reshape(1),
                               (numpy.array([distv]) / dodge_config.HEIGHT * 2 - 1).reshape(1),
                               (numpy.array([righty]) / dodge_config.HEIGHT * 2 - 1).reshape(dodge_config.OBJECTSHOR),
                               (numpy.array([lefty]) / dodge_config.HEIGHT * 2 - 1).reshape(dodge_config.OBJECTSHOR),
                               (numpy.array([downx]) / dodge_config.WIDTH * 2 - 1).reshape(dodge_config.OBJECTSVER),
                               (numpy.array([upx]) / dodge_config.WIDTH * 2 - 1).reshape(dodge_config.OBJECTSVER),
                               (numpy.array([presentx]) / dodge_config.WIDTH * 2 - 1).reshape(1),
                               (numpy.array([presenty]) / dodge_config.HEIGHT * 2 - 1).reshape(1)
                               ))).reshape((1, dodge_config.OBS_SIZE))


class PresentHunterUI(GameUI):
    def __init__(self):
        super().__init__(ph_config.WIDTH, ph_config.HEIGHT)

        self.backgroundTimer = 0

        # image creation
        self.img_player = pygame.image.load("resources/character.png")
        self.img_present = pygame.image.load("resources/present.png")

        self.player_trans = pygame.Surface((ph_config.PLAYERWIDTH, ph_config.PLAYERWIDTH))
        self.player_trans.set_alpha(100)
        self.player_trans.fill((0, 0, 0))

        self.present_trans = pygame.Surface((ph_config.PRESENTWIDTH, ph_config.PRESENTHEIGHT))
        self.present_trans.set_alpha(100)
        self.present_trans.fill((0, 150, 0))

        # mouse selection
        self.selected = -1
        self.override_action = None

    def draw(self, window, actor, font, game):
        playerx, playery, presentx, presenty = self.decode_obs(game.obs)
        reward = game.rewards().any()
        if reward:
            window.fill((255, 255, 255))
        else:
            window.fill((int(255 * 0.5 * (1 + math.sin(0.001 * self.backgroundTimer))),
                         int(255 * 0.5 * (1 + math.sin(0.001 * self.backgroundTimer + 2 * math.pi / 3))),
                         int(255 * 0.5 * (1 + math.sin(0.001 * self.backgroundTimer + 4 * math.pi / 3)))))

        window.blit(self.img_present, (presentx - ph_config.PRESENTWIDTH / 2, presenty - ph_config.PRESENTHEIGHT / 2))

        window.blit(self.img_player, (playerx - ph_config.PLAYERWIDTH / 2, playery - ph_config.PLAYERWIDTH / 2))

        if actor.is_agent:
            with torch.no_grad():
                r, e, _, _ = actor.agent((game.obs, None))
                r = r.cpu()
                e = e.cpu()
            for i in range(r.shape[1]):
                text_surface = font.render('Rew: {:.4f}, Est: {:.4f}'.format(r.numpy()[0, i], e.numpy()[0, i]), False, (0, 0, 0))
                window.blit(text_surface, (0, 50 * i))

    def update_pause(self, actor, game, user_input):
        left, right, wheel = pygame.mouse.get_pressed()
        if user_input.mouse_left:
            self.selected = self.get_picked(user_input.mouse_x, user_input.mouse_y, game.obs)
            self.override_action = None
        if self.selected >= 0:
            if left:
                game.obs = self.move_observation(user_input.mouse_x, user_input.mouse_y, game.obs)
            else:
                self.selected = -1
        else:
            # if no object is clicked change player action
            if left:
                playerx, playery, _, _ = self.decode_obs(game.obs)
                act_x = (user_input.mouse_x - playerx) / (ph_config.PLAYERSPEED * ph_config.ACT_DRAW_LENGTH)
                act_y = (user_input.mouse_y - playery) / (ph_config.PLAYERSPEED * ph_config.ACT_DRAW_LENGTH)

                self.override_action = torch.FloatTensor().new_tensor([act_x, act_y]).reshape(1, 2)

    def draw_pause(self, window, actor, font, game):
        if actor.is_agent:
            playerx, playery, presentx, presenty = self.decode_obs(game.obs)

            act = torch.FloatTensor().new_zeros(1, 2)
            grad = actor.agent.reward_gradient(game.obs, torch.FloatTensor().new_tensor([-1, .5])).cpu()
            grad_fac = .1

            obs_shadow = game.obs + grad * grad_fac
            f_playerx, f_playery, f_presentx, f_presenty = \
                self.decode_obs(obs_shadow)

            window.blit(self.present_trans, (f_presentx - ph_config.PRESENTWIDTH / 2, f_presenty - ph_config.PRESENTHEIGHT / 2))
            window.blit(self.player_trans, (f_playerx - ph_config.PLAYERWIDTH / 2, f_playery - ph_config.PLAYERWIDTH / 2))

    # 0: player, 1: present
    def get_picked(self, x, y, obs):
        playerx, playery, presentx, presenty = self.decode_obs(obs)
        if util.is_in_rect(x, y, playerx, playery, ph_config.PLAYERWIDTH, ph_config.PLAYERWIDTH):
            return 0
        if util.is_in_rect(x, y, presentx, presenty, ph_config.PRESENTWIDTH, ph_config.PRESENTHEIGHT):
            return 1
        return -1

    def move_observation(self, mouse_x, mouse_y, obs):
        playerx, playery, presentx, presenty = self.decode_obs(obs)

        if self.selected < 0:
            return
        if self.selected == 0:
            playerx = mouse_x
            playery = mouse_y
        elif self.selected == 1:
            presentx = mouse_x
            presenty = mouse_y

        return self.encode_obs((playerx, playery, presentx, presenty))

    @staticmethod
    def decode_obs(obs):
        playerx = (obs[0, 0].numpy() + 1) / 2 * ph_config.WIDTH
        playery = (obs[0, 1].numpy() + 1) / 2 * ph_config.HEIGHT
        presentx = (obs[0, 2].numpy() + 1) / 2 * ph_config.WIDTH
        presenty = (obs[0, 3].numpy() + 1) / 2 * ph_config.HEIGHT

        return playerx, playery, presentx, presenty

    @staticmethod
    def encode_obs(values):
        playerx, playery, presentx, presenty = values

        return torch.FloatTensor().new_tensor(
            numpy.concatenate(((numpy.array([playerx]) / ph_config.WIDTH * 2 - 1).reshape(1),
                               (numpy.array([playery]) / ph_config.HEIGHT * 2 - 1).reshape(1),
                               (numpy.array([presentx]) / ph_config.WIDTH * 2 - 1).reshape(1),
                               (numpy.array([presenty]) / ph_config.HEIGHT * 2 - 1).reshape(1)
                               ))).reshape((1, 4))
