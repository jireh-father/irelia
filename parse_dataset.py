import json
import sys
from util import common
from game.game import Game
import tensorflow as tf
import numpy as np
from util.dataset import Dataset
import os

FLAGS = tf.app.flags.FLAGS

common.set_flags()

# f = open(os.path.join(FLAGS.dataset_dir, "korean-chess-records-dataset.txt"))
f = open("records.txt")
position = {"masangmasang": 0, "masangsangma": 1, "sangmasangma": 2, "sangmamasang": 3}
ds = Dataset()
# ds.open(os.path.join(FLAGS.dataset_dir, "dataset.csv"))
ds.open("dataset.csv")
for n, line in enumerate(f):
    print(n)
    data = json.loads(line)
    actions = data["records"]
    winner = data["winner"]
    red_position = data["red_position_type"]
    blue_position = data["blue_position_type"]
    if red_position not in position or blue_position not in position:
        sys.exit("invalid position!!")

    env = Game.make("KoreanChess-v1",
                    {"position_type": [position[blue_position], position[red_position]]})
    first_state = env.reset()
    info = {"over_limit_step": False, "is_draw": False, "winner": winner}
    state, _ = env.encode_state(first_state)
    state_history = [first_state.tolist()]
    mcts_history = []
    for i, action in enumerate(actions):
        turn = "r" if i % 2 == 0 else "b"
        state[action["to_y"]][action["to_x"]] = state[action["y"]][action["x"]]
        state[action["y"]][action["x"]] = 0
        state_history.append(env.decode_state(state, turn).tolist())
        policy_probs = np.array([.0] * 90)
        action["from_x"] = action["x"]
        action["from_y"] = action["y"]
        action = env.encode_action(action)
        policy_probs[action[0]] = 0.5
        policy_probs[action[1]] = 0.5
        mcts_history.append(policy_probs.tolist())
    del state_history[-1]
    ds.write(info, state_history, mcts_history)
ds.close()
