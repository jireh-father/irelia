# coding=utf8
import tensorflow as tf
from game.game import Game
from core.greedy import Greedy
from util import common
from util import user_input
import traceback
import time
from core.mcts_uct import MctsUct

FLAGS = tf.app.flags.FLAGS

common.set_flags()

env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": FLAGS.max_step,
                                   "print_mcts_history": FLAGS.print_mcts_history,
                                   "use_color_print": FLAGS.use_color_print})
state = env.reset()
i = 0
user_action_idx = -1

mcts = MctsUct(env, FLAGS.max_simulation)
while True:
    if i % 2 == 0:
        from_x, from_y, to_x, to_y = user_input.get_user_input()

        try:
            legal_actions = env.get_all_actions()
            user_action = {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y}
            state, reward, done, _ = env.step(user_action)

            for j, legal_action in enumerate(legal_actions):
                if legal_action == user_action:
                    user_action_idx = j
                    break
            if done:
                print("User win")
                break

        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
    else:
        action = mcts.search(state, env.current_turn)

        state, reward, done, info = env.step(action)
        if done:
            break
    i += 1
