# coding=utf8
import tensorflow as tf
from game.game import Game
from core.mcts_random import Mcts
from util import common
from util import user_input
import traceback
import time

FLAGS = tf.app.flags.FLAGS

common.set_flags()
tf.app.flags.DEFINE_integer('max_rollouts', 20, "exploration step")

env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": FLAGS.max_step,
                                   "print_mcts_history": FLAGS.print_mcts_history,
                                   "use_color_print": FLAGS.use_color_print})
state = env.reset()
i = 0
user_action_idx = -1
mcts = Mcts(state, env, FLAGS.max_simulation, max_rollout=FLAGS.max_rollouts)
while True:
    if i % 2 == 0:
        from_x, from_y, to_x, to_y = user_input.get_user_input()

        try:
            legal_actions = env.get_all_actions()
            user_action = {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y}
            new_state, reward, done, _ = env.step(user_action)

            for j, legal_action in enumerate(legal_actions):
                if legal_action == user_action:
                    user_action_idx = j
                    break
            if done:
                print("The End")
                break

        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
    else:
        start_time = time.time()
        mcts_action = mcts.search(0, user_action_idx)
        print("elased time : %f" % (time.time() - start_time))
        if FLAGS.print_mcts_tree:
            mcts.print_tree()
        try:
            new_state, reward, done, _ = env.step(mcts_action)
            if done:
                print("The End")
                break
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
    i += 1
