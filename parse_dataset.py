import json
import sys
from util import common
from game.game import Game
from game import korean_chess_util
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
error = 0
nn =0
for n, line in enumerate(f):

    print(n)
    if position[red_position] == 2 and position[blue_position] == 0:
        nn += 1
    if nn == 10:
        break
    data = json.loads(line)
    actions = data["records"]
    winner = data["winner"]
    red_position = data["red_position_type"]
    blue_position = data["blue_position_type"]
    print(position[red_position], position[blue_position])
    sys.exit()
    if red_position not in position or blue_position not in position:
        sys.exit("invalid position!!")

    env = Game.make("KoreanChess-v1",
                    {"position_type": [position[blue_position], position[red_position]], "limit_repeat": 6})
    first_state = env.reset()
    info = {"over_limit_step": False, "is_draw": False, "winner": winner}
    state, _ = env.encode_state(first_state)
    state_history = [first_state.tolist()]
    mcts_history = []
    for i, action in enumerate(actions):
        action["from_x"] = action["x"]
        action["from_y"] = action["y"]
        if i % 2 == 1:
            [action] = korean_chess_util.reverse_actions([action])
        turn = "r" if i % 2 == 0 else "b"

        state[action["to_y"]][action["to_x"]] = state[action["from_y"]][action["from_x"]]
        state[action["from_y"]][action["from_x"]] = 0

        print(i, action)
        first_state, reward, done, info = env.step(action)
        if info is False:
            break
        state_history.append(env.decode_state(state, turn).tolist())
        # policy_probs = np.array([.0] * 180)
        policy_probs = np.array([.0] * 90)

        action = env.encode_action(action)
        # policy_probs[action[0]] = 0.5
        # policy_probs[action[1]] = 0.5
        policy_probs[action[0]] = 1.
        policy_probs[action[1]] = 1.
        # policy_probs[action[1] + 90] = 0.5
        mcts_history.append(policy_probs.tolist())
        # break
    del state_history[-1]
    if info:
        ds.write(info, state_history, mcts_history)
    # break
        # ds.close()
        # sys.exit()
print(error)
ds.close()
