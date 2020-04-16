from abstract_game import AbstractGame, Actor
import pygame
import numpy
import util
import torch
import math


class DodgeGame(AbstractGame):

    def __init__(self, nn_time_step=60, frame_rate=121, estimator=None):
        super().__init__(2, 30, 2, util.WIDTH, util.HEIGHT, nn_time_step, frame_rate)

        self.estimator = estimator

        self.backgroundTimer = 0

        # image creation
        self.img_player = util.load_image("character.png")
        self.img_obstacle = util.load_image("obstacle.png")
        self.img_present = util.load_image("present.png")

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

    def draw(self, window, actor, font):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(self.observation)

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

        super().draw(window, actor, font)

    def update_pause(self, actor):
        left, right, wheel = pygame.mouse.get_pressed()
        if self.user_input.mouse_left:
            self.selected = self.get_picked(self.user_input.mouse_x, self.user_input.mouse_y)
            self.override_action = None
        if self.selected >= 0:
            if left:
                self.move_observation(self.user_input.mouse_x, self.user_input.mouse_y)
            else:
                self.selected = -1
        else:
            # if no object is clicked change player action
            if left:
                playerx, playery, _, _, _, _, _, _, _, _ = self.decode_obs(self.observation)
                act_x = (self.user_input.mouse_x - playerx) / (util.PLAYERSPEED * util.ACT_DRAW_LENGTH)
                act_y = (self.user_input.mouse_y - playery) / (util.PLAYERSPEED * util.ACT_DRAW_LENGTH)

                self.override_action = torch.FloatTensor().new_tensor([act_x, act_y]).reshape(1, 2)

    def draw_pause(self, window, actor, font):
        if actor.is_agent:
            playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(
                self.observation)
            act = torch.FloatTensor().new_zeros(1, 2)
            grad = actor.agent.reward_gradient(self.observation, torch.FloatTensor().new_tensor([-1, .5]))
            grad_fac = .1

            obs_shadow = self.observation + grad * grad_fac
            f_playerx, f_playery, f_disth, f_distv, f_righty, f_lefty, f_downx, f_upx, f_presentx, f_presenty = self.decode_obs(
                obs_shadow)

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

    def update(self, time_elapsed, speed, actor, run_in_background, track_events):
        super().update(time_elapsed, speed, actor, run_in_background, track_events)
        if not run_in_background:
            self.backgroundTimer += time_elapsed

    def run(self, actor, max_time=100000, speed=.5,
            run_in_background=False, track_events=True, overwrite=True, filename="./data.csv", console=True,
            initial_obs=None, start_paused=False):
        return super().run(actor, max_time, speed, run_in_background, track_events, overwrite, filename=filename,
                           console=console, initial_obs=initial_obs, start_paused=start_paused)

    def decode_obs(self, obs):
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

    def encode_obs(self, values):
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

    def initial_obs(self):
        playerx = (numpy.random.random(size=1) * .5 + .25) * util.WIDTH
        playery = (numpy.random.random(size=1) * .5 + .25) * util.HEIGHT
        disth = numpy.array([0.])
        distv = numpy.array([0.])
        righty = numpy.random.random(size=5) * util.HEIGHT
        lefty = numpy.random.random(size=5) * util.HEIGHT
        downx = numpy.random.random(size=7) * util.WIDTH
        upx = numpy.random.random(size=7) * util.WIDTH
        presentx = numpy.random.random(size=1) * util.WIDTH
        presenty = numpy.random.random(size=1) * util.HEIGHT

        return self.encode_obs((playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty))

    def step_obs(self, time_elapsed, action, speed, is_nn_time_step):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(self.observation)
        actionx = action[0, 0].numpy()
        actiony = action[0, 1].numpy()

        if is_nn_time_step and util.rect_collide(playerx, playery, presentx, presenty, util.PLAYERWIDTH,
                                                 util.PLAYERWIDTH, util.PRESENTWIDTH, util.PRESENTHEIGHT):
            presentx = numpy.random.random(size=1) * util.WIDTH
            presenty = numpy.random.random(size=1) * util.HEIGHT

        playerx += actionx * util.PLAYERSPEED * time_elapsed * speed
        playery += actiony * util.PLAYERSPEED * time_elapsed * speed
        distv += 1 * util.OBSTACLESPEED * time_elapsed * speed
        disth += 1 * util.OBSTACLESPEED * time_elapsed * speed

        if playerx <= util.PLAYERWIDTH / 2:
            playerx = util.PLAYERWIDTH / 2
        elif playerx >= util.WIDTH - util.PLAYERWIDTH / 2:
            playerx = util.WIDTH - util.PLAYERWIDTH / 2
        if playery <= util.PLAYERWIDTH / 2:
            playery = util.PLAYERWIDTH / 2
        elif playery >= util.HEIGHT - util.PLAYERWIDTH / 2:
            playery = util.HEIGHT - util.PLAYERWIDTH / 2

        if disth >= util.WIDTH - util.OBSTACLEWIDTH:
            disth = numpy.array([0.])
            righty = numpy.random.random(size=5) * util.HEIGHT
            lefty = numpy.random.random(size=5) * util.HEIGHT
        if distv >= util.HEIGHT - util.OBSTACLEWIDTH:
            distv = numpy.array([0.])
            downx = numpy.random.random(size=7) * util.WIDTH
            upx = numpy.random.random(size=7) * util.WIDTH

        self.observation = self.encode_obs((playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty))

    def receive_reward(self):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(self.observation)

        collided = False
        for i in range(5):
            if util.rect_collide(playerx, playery, disth, righty[i], util.PLAYERWIDTH, util.PLAYERWIDTH,
                                 util.OBSTACLEWIDTH, util.OBSTACLEWIDTH):
                collided = True
        for i in range(5):
            if util.rect_collide(playerx, playery, util.WIDTH - disth, lefty[i], util.PLAYERWIDTH, util.PLAYERWIDTH,
                                 util.OBSTACLEWIDTH, util.OBSTACLEWIDTH):
                collided = True
        for i in range(7):
            if util.rect_collide(playerx, playery, downx[i], distv, util.PLAYERWIDTH, util.PLAYERWIDTH,
                                 util.OBSTACLEWIDTH, util.OBSTACLEWIDTH):
                collided = True
        for i in range(7):
            if util.rect_collide(playerx, playery, upx[i], util.HEIGHT - distv, util.PLAYERWIDTH, util.PLAYERWIDTH,
                                 util.OBSTACLEWIDTH, util.OBSTACLEWIDTH):
                collided = True

        present = False
        if util.rect_collide(playerx, playery, presentx, presenty, util.PLAYERWIDTH, util.PLAYERWIDTH,
                             util.PRESENTWIDTH, util.PRESENTHEIGHT):
            present = True

        return torch.BoolTensor().new_tensor([collided, present])

    # 0: player, 12345: r, 678910: l 1234567: d, 8901234: u, 25: present
    def get_picked(self, x, y):
        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(self.observation)
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

    def move_observation(self, mouse_x, mouse_y):

        playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty = self.decode_obs(self.observation)

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

        self.observation = self.encode_obs((playerx, playery, disth, distv, righty, lefty, downx, upx, presentx, presenty))


class ActorHuman(Actor):
    def __init__(self):
        super().__init__(False)

    def get_action(self, observation, user_input, nn_step):
        x = 1 if user_input.right else (-1 if user_input.left else 0)
        y = 1 if user_input.down else (-1 if user_input.up else 0)
        return torch.FloatTensor().new_tensor([x, y]).reshape((1, 2))
