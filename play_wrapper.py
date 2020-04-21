import pygame

import dodge_parallel
import dodge_ui
import user_input


class PlayWrapper:
    def __init__(self,
                 game: dodge_parallel.GameParallel,
                 ui: dodge_ui.GameUI,
                 width=1200, height=675, frame_rate=121):

        self.game = game
        self.ui = ui

        self.frame_rate = frame_rate

        self.width = width
        self.height = height

        self.user_input = user_input.UserInput()
        self.pause = False

    def run(self, actor, max_time, speed, initial_obs=None, start_paused=False):
        window = pygame.display.set_mode((self.width, self.height))
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
