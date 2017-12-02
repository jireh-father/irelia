from util import common
import sys, traceback
from core.mcts import Mcts


def self_play(env, model, max_simulation, max_step, c_puct, exploration_step, reuse_mcts=True, print_mcts_tree=False,
              num_state_history=7):
    state = env.reset()

    mcts = Mcts(state, env, model, max_simulation=max_simulation, c_puct=c_puct, num_state_history=num_state_history)
    state_history = [state.tolist()]
    mcts_history = []
    temperature = 1
    info = None
    step = 0
    old_action_idx = None
    while step <= max_step:
        # for step in range(max_step):
        common.log("step: %d" % step)
        if step >= exploration_step:
            common.log("temperature down")
            temperature = 0
        actions = env.get_all_actions()
        action_probs = mcts.search(temperature, [] if old_action_idx is None else [old_action_idx])
        action_idx = mcts.get_action_idx(action_probs)
        action = actions[action_idx]

        if print_mcts_tree:
            mcts.print_tree()
        try:
            state, reward, done, info = env.step(action)
            if reward is False:
                print("repeat!!")
                if len(action_probs) == 1:
                    info["winner"] = env.next_turn
                    break
                else:
                    action_probs[action_idx] = action_probs.min()
                    action_probs = action_probs / action_probs.sum()
                    second_action_idx = action_idx
                    while action_idx == second_action_idx:
                        second_action_idx = mcts.get_action_idx(action_probs)
                    action_idx = second_action_idx
                    print("retry second action for repeating %d" % action_idx)
                    action = actions[action_idx]
                    state, reward, done, info = env.step(action)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            continue

        if len(actions) != len(action_probs):
            print(len(actions), len(action_probs))
            sys.exit("error!!! action count!!")

        if reuse_mcts:
            mcts = Mcts(state, env, model, max_simulation=max_simulation, c_puct=c_puct)
        mcts_history.append(env.convert_action_probs_to_policy_probs(actions, action_probs))

        old_action_idx = action_idx

        step += 1
        if done:
            break
        state_history.append(state.tolist())
    return info, state_history, mcts_history


def eval_play(env, blue_model, red_model, max_simulation, max_step, c_puct, reuse_mcts=True, print_mcts_tree=False,
              num_state_history=7):
    state = env.reset()
    blue_mcts = Mcts(state, env, blue_model, max_simulation=max_simulation, c_puct=c_puct,
                     num_state_history=num_state_history)
    red_mcts = Mcts(state, env, red_model, max_simulation=max_simulation, c_puct=c_puct,
                    num_state_history=num_state_history)
    temperature = 0
    info = None
    step = 0
    action_idx_history = []
    mcts = blue_mcts
    while step <= max_step:
        # for step in range(max_step):
        common.log("step: %d" % step)
        actions = env.get_all_actions()
        action_probs = mcts.search(temperature, [] if not action_idx_history else action_idx_history[-2:])
        action_idx = mcts.get_action_idx(action_probs)
        action = actions[action_idx]

        if print_mcts_tree:
            mcts.print_tree()
        try:
            state, reward, done, info = env.step(action)
            if reward is False:
                print("repeat!!")
                if len(action_probs) == 1:
                    info["winner"] = env.next_turn
                    break
                else:
                    action_probs[action_idx] = action_probs.min()
                    action_probs = action_probs / action_probs.sum()
                    second_action_idx = action_idx
                    while action_idx == second_action_idx:
                        second_action_idx = mcts.get_action_idx(action_probs)
                    action_idx = second_action_idx
                    print("retry second action for repeating %d" % action_idx)
                    action = actions[action_idx]
                    state, reward, done, info = env.step(action)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            continue

        if len(actions) != len(action_probs):
            print(len(actions), len(action_probs))
            sys.exit("error!!! action count!!")

        action_idx_history.append(action_idx)

        if reuse_mcts:
            blue_mcts = Mcts(state, env, blue_model, max_simulation=max_simulation, c_puct=c_puct,
                             num_state_history=num_state_history)
            red_mcts = Mcts(state, env, red_model, max_simulation=max_simulation, c_puct=c_puct,
                            num_state_history=num_state_history)

        if step % 2 == 0:
            mcts = red_mcts
        else:
            mcts = blue_mcts

        step += 1

        if done:
            break
    return info