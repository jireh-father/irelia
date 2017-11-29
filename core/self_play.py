from util import common
import sys, traceback
from core.mcts import Mcts


def self_play(env, model, max_simulation, max_step, c_puct, exploration_step, reuse_mcts=True,
              print_mcts_tree=False):
    state = env.reset()

    mcts = Mcts(state, env, model, max_simulation=max_simulation, c_puct=c_puct)
    state_history = [state.tolist()]
    mcts_history = []
    temperature = 1
    info = None
    step = 0
    while step <= max_step:
        # for step in range(max_step):
        common.log("step: %d" % step)
        if step >= exploration_step:
            common.log("temperature down")
            temperature = 0
        actions = env.get_all_actions()
        search_action_probs, action = mcts.search(temperature)
        if print_mcts_tree:
            mcts.print_tree()
        try:
            state, reward, done, info = env.step(action)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            continue

        if len(actions) != len(search_action_probs):
            print(len(actions), len(search_action_probs))
            sys.exit("error!!! action count!!")

        if reuse_mcts:
            mcts = Mcts(state, env, model, max_simulation=max_simulation, c_puct=c_puct)
        mcts_history.append(env.convert_action_probs_to_policy_probs(actions, search_action_probs))

        step += 1
        if done:
            break
        state_history.append(state.tolist())
    return info, state_history, mcts_history
