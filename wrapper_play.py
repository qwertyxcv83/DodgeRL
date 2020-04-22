import pygame

import game_parallel
import game_ui


class WrapperPlay:
    def __init__(self,
                 game: game_parallel.GameParallel,
                 ui: game_ui.GameUI,
                 frame_rate=121):

        self.game = game
        self.ui = ui

        self.frame_rate = frame_rate

        self.user_input = UserInput()
        self.pause = False

    def run(self, actor, max_time, speed, initial_obs=None, start_paused=False):
        window = pygame.display.set_mode((self.ui.width, self.ui.height))
        pygame.font.init()
        font = pygame.font.SysFont('Verdana', 20)
        clock = pygame.time.Clock()

        self.user_input.reset()
        self.pause = start_paused

        run = True
        total_time = 0

        self.game.obs = self.game.initial_obs() if initial_obs is None else initial_obs

        while run:
            time_elapsed = clock.tick(self.frame_rate)
            total_time += time_elapsed

            self.user_input.set_user_input(pygame.event.get())

            self.pause = self.user_input.space ^ self.pause

            run = self.user_input.run and total_time < max_time

            if self.pause:
                self.game.obs = self.ui.update_pause(actor, self.game.obs, self.user_input)
            else:
                action = actor.get_action(self.game.obs, self.user_input, True)

                self.game.action_move(action, time_elapsed, speed)
                self.game.action_triggered()

            self.ui.draw(window, actor, font, self.game.obs)
            if self.pause:
                self.ui.draw_pause(window, actor, font, self.game.obs)

            pygame.display.update()

        pygame.quit()


class UserInput:
    def __init__(self):
        self.run = True
        self.up = False
        self.down = False
        self.right = False
        self.left = False
        self.space = False
        self.mouse_left = False  # only true if pressed this tick
        self.mouse_right = False  # only true if pressed this tick
        self.mouse_wheel = False  # only true if pressed this tick
        self.mouse_x = 0
        self.mouse_y = 0

    def set_user_input(self, events):
        key_down = False
        for event in events:
            if event.type == pygame.QUIT:
                self.run = False
            if event.type == pygame.KEYDOWN:
                key_down = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_left, self.mouse_right, self.mouse_wheel = pygame.mouse.get_pressed()
            else:
                self.mouse_left = False
                self.mouse_right = False
                self.mouse_wheel = False

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            self.up = True
            self.down = False
            self.right = False
            self.left = False
        if pressed[pygame.K_DOWN]:
            self.up = False
            self.down = True
            self.right = False
            self.left = False
        if pressed[pygame.K_RIGHT]:
            self.up = False
            self.down = False
            self.right = True
            self.left = False
        if pressed[pygame.K_LEFT]:
            self.up = False
            self.down = False
            self.right = False
            self.left = True
        if pressed[pygame.K_SPACE] and key_down:
            self.space = True
        else:
            self.space = False
        self.mouse_x, self.mouse_y = pygame.mouse.get_pos()

    def reset(self):
        self.run = True
        self.up = False
        self.down = False
        self.right = False
        self.left = False
        self.space = False
        self.mouse_left = False
        self.mouse_right = False
        self.mouse_wheel = False
        self.mouse_x = 0
        self.mouse_y = 0
