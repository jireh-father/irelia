from util import common
import sys, traceback
from core.mcts import Mcts
from core.mcts_reward import Mcts as Mcts_reward
from core.mcts_with_reward import Mcts as MctsWithReward


def self_play(env, model, max_simulation, max_step, c_puct, exploration_step, reuse_mcts=True, print_mcts_tree=False,
              num_state_history=7, print_mcts_search=True, use_reward_mcts=False, begin_temperature=1):
    state = env.reset()
    MctsClass = Mcts
    if use_reward_mcts:
        MctsClass = Mcts_reward

    mcts = MctsClass(state, env, model, max_simulation=max_simulation, c_puct=c_puct,
                     num_state_history=num_state_history,
                     print_mcts_search=print_mcts_search)
    state_history = [state.tolist()]
    mcts_history = []
    temperature = begin_temperature
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
        if not reuse_mcts:
            action_probs = mcts.search(temperature)
        else:
            action_probs = mcts.search(temperature, [] if old_action_idx is None else [old_action_idx])
        if len(actions) != len(action_probs):
            print("legal actions", len(actions), "mcts actions", len(action_probs))
            print("legal state")
            env.print_env()
            sys.exit("error!!! action count!!")
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

        if not reuse_mcts:
            mcts = MctsClass(state, env, model, max_simulation=max_simulation, c_puct=c_puct, init_root_edges=True)
        mcts_history.append(env.convert_action_probs_to_policy_probs(actions, action_probs))

        old_action_idx = action_idx

        step += 1
        if done:
            break
        state_history.append(state.tolist())
    return info, state_history, mcts_history


def self_play_only_net(env, model, max_step):
    state = env.reset()
    temperature = 0
    info = None
    step = 0
    state_history = [state]
    while step <= max_step:
        # for step in range(max_step):
        common.log("step: %d" % step)
        actions = env.get_all_actions()
        converted_state = common.convert_state_history_to_model_input(state_history[-8:])
        policy, value = model.inference(converted_state)
        print("value %f" % value)
        print("policy", policy)

        action_probs = model.filter_action_probs(policy, actions, env)
        print("action probs!", action_probs)

        if len(actions) != len(action_probs):
            print(len(actions), len(action_probs))
            sys.exit("error!!! action count!!")

        action_idx = model.get_action_idx(action_probs, temperature)
        action = actions[action_idx]

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
                        second_action_idx = model.get_action_idx(action_probs, temperature)
                    action_idx = second_action_idx
                    print("retry second action for repeating %d" % action_idx)
                    action = actions[action_idx]
                    state, reward, done, info = env.step(action)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            continue

        state_history.append(state)

        step += 1
        if done:
            break
    return info


def self_play_mcts_with_reward(env, model, max_simulation, max_step, c_puct, exploration_step, reuse_mcts=True,
                               print_mcts_tree=False,
                               num_state_history=7, print_mcts_search=True, begin_temperature=1, reward_ratio=1.):
    state = env.reset()

    mcts = MctsWithReward(state, env, model, max_simulation=max_simulation, c_puct=c_puct,
                          num_state_history=num_state_history,
                          print_mcts_search=print_mcts_search, reward_ratio=reward_ratio)
    state_history = [state.tolist()]
    mcts_history = []
    temperature = begin_temperature
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
        if not reuse_mcts:
            action_probs = mcts.search(temperature)
        else:
            action_probs = mcts.search(temperature, [] if old_action_idx is None else [old_action_idx])
        if len(actions) != len(action_probs):
            print("legal actions", len(actions), "mcts actions", len(action_probs))
            print("legal state")
            env.print_env()
            sys.exit("error!!! action count!!")
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

        mcts.print_line(action_idx)

        if not reuse_mcts:
            mcts = MctsWithReward(state, env, model, max_simulation=max_simulation, c_puct=c_puct, init_root_edges=True)
        mcts_history.append(env.convert_action_probs_to_policy_probs(actions, action_probs))

        old_action_idx = action_idx

        step += 1
        if done:
            break
        state_history.append(state.tolist())
    return info, state_history, mcts_history


def eval_play(env, blue_model, red_model, max_simulation, max_step, c_puct, reuse_mcts=True, print_mcts_tree=False,
              num_state_history=7, print_mcts_search=False):
    state = env.reset()
    blue_mcts = Mcts(state, env, blue_model, max_simulation=max_simulation, c_puct=c_puct,
                     num_state_history=num_state_history, print_mcts_search=print_mcts_search, init_root_edges=True)
    red_mcts = Mcts(state, env, red_model, max_simulation=max_simulation, c_puct=c_puct,
                    num_state_history=num_state_history, print_mcts_search=print_mcts_search, init_root_edges=True)
    temperature = 0
    info = None
    step = 0
    action_idx_history = []
    mcts = blue_mcts
    while step <= max_step:
        # for step in range(max_step):
        common.log("step: %d" % step)
        actions = env.get_all_actions()
        if not reuse_mcts:
            action_probs = mcts.search(temperature)
        else:
            action_probs = mcts.search(temperature, [] if not action_idx_history else action_idx_history[-2:])

        if len(actions) != len(action_probs):
            print(len(actions), len(action_probs))
            sys.exit("error!!! action count!!")

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

        print("action_idx", action_idx)
        action_idx_history.append(action_idx)

        if not reuse_mcts:
            blue_mcts = Mcts(state, env, blue_model, max_simulation=max_simulation, c_puct=c_puct,
                             num_state_history=num_state_history, print_mcts_search=print_mcts_search,
                             init_root_edges=True)
            red_mcts = Mcts(state, env, red_model, max_simulation=max_simulation, c_puct=c_puct,
                            num_state_history=num_state_history, print_mcts_search=print_mcts_search,
                            init_root_edges=True)

        if step % 2 == 0:
            mcts = red_mcts
        else:
            mcts = blue_mcts

        step += 1

        if done:
            break
    return info
