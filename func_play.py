import game_ui
import game_parallel
import func_learn
import wrapper_play
import actors
import rl_net


def run_fast(path="./agent.pth", secs=60, speed=.4):
    agent = func_learn.load(rl_net.ModelAgent(30, 2, 2, False), path, 'cpu')
    actor = actors.ActorNN(agent, noise=0)
    physics = game_parallel.DodgeParallel()
    ui = game_ui.DodgeUI()
    wrap = wrapper_play.WrapperPlay(physics, ui)
    wrap.run(actor, secs*1000, speed)
