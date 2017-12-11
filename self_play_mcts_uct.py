from core import play
import tensorflow as tf
from util import common
from util.dataset import Dataset
import os
from core.model import Model
from game.game import Game
import uuid
from util.common import log
from core.mcts_uct import MctsUct

FLAGS = tf.app.flags.FLAGS

common.set_flags()

env = Game.make("KoreanChess-v1",
                {"use_check": False, "limit_step": FLAGS.max_step, "use_color_print": FLAGS.use_color_print,
                 "use_cache": FLAGS.use_cache})

mcts = MctsUct(env, FLAGS.max_simulation)
state = env.reset()
while True:
    """"""
    """self-play"""
    action = mcts.search(state, env.current_turn)

    state, reward, done, info = env.step(action)
    if done:
        break

print("winner", info["winner"])
