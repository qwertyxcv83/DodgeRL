import game_ui
import game_parallel
import func_learn
import wrapper_play
import actors
import rl_net
import dodge_config
import ph_config


def run_fast(path="./agent.pth", secs=60, speed=.4, parallel=game_parallel.DodgeParallel, g_ui=game_ui.DodgeUI, n_hidden=200):
    agent = rl_net.ModelAgent(dodge_config.OBS_SIZE, dodge_config.ACTION_SIZE, [-1, 1], False, n_hidden=n_hidden)
    agent = func_learn.load(agent, path, 'cpu')
    actor = actors.ActorNN(agent, noise=0)
    physics = parallel()
    ui = g_ui()
    wrap = wrapper_play.WrapperPlay(physics, ui)
    wrap.run(actor, secs*1000, speed)


def run_new(secs=60, speed=.4, parallel=game_parallel.DodgeParallel, g_ui=game_ui.DodgeUI):
    agent = rl_net.ModelAgent(dodge_config.OBS_SIZE, dodge_config.ACTION_SIZE, [-1, 1], False)
    actor = actors.ActorNN(agent, noise=0)
    physics = parallel()
    ui = g_ui()
    wrap = wrapper_play.WrapperPlay(physics, ui)
    wrap.run(actor, secs * 1000, speed)


def play(secs=60, speed=.4, parallel=game_parallel.DodgeParallel, g_ui=game_ui.DodgeUI):
    actor = actors.ActorDodgeHuman()
    physics = parallel()
    ui = g_ui()
    wrap = wrapper_play.WrapperPlay(physics, ui)
    wrap.run(actor, secs * 1000, speed)
