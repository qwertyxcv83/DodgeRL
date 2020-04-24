import game_ui
import game_parallel
import func_learn
import wrapper_play
import actors
import rl_net


def run_fast(path="./agent.pth", secs=60, speed=.4, parallel=game_parallel.DodgeParallel, g_ui=game_ui.DodgeUI):
    agent = func_learn.load(rl_net.ModelAgent(4, 2, [1], False, n_hidden=20), path, 'cpu')
    actor = actors.ActorNN(agent, noise=0)
    physics = parallel()
    ui = g_ui()
    wrap = wrapper_play.WrapperPlay(physics, ui)
    wrap.run(actor, secs*1000, speed)


def run_new(secs=60, speed=.4, parallel=game_parallel.DodgeParallel, g_ui=game_ui.DodgeUI):
    agent = rl_net.ModelAgent(4, 2, [1], False)
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
