from util.dataset2 import Dataset
import tensorflow as tf
from util import common
import numpy as np
from game.game import Game


def filter_action_probs(action_probs, action_probs2, legal_actions, env):
    legal_action_probs = []
    for legal_action in legal_actions:
        legal_action = env.encode_action(legal_action)
        legal_action_probs.append(action_probs[legal_action[0]] + action_probs2[legal_action[1]])

    legal_action_probs = np.array(legal_action_probs)
    if (legal_action_probs == 0).all():
        legal_action_probs = np.array([1. / len(legal_action_probs)] * len(legal_action_probs))
    else:
        legal_action_probs = legal_action_probs / legal_action_probs.sum()
    return legal_action_probs


FLAGS = tf.app.flags.FLAGS

common.set_flags()
tf.app.flags.DEFINE_string('dataset_path', None, "dataset_dir")

env = Game.make("KoreanChess-v1",
                {"use_check": False, "limit_step": FLAGS.max_step, "print_mcts_history": True,
                 "use_color_print": True})
env.reset()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
ds = Dataset(sess)
FLAGS.dataset_path = "dataset.csv"
ds.make_dataset([FLAGS.dataset_path], FLAGS.batch_size, shuffle_buffer_size=0)

ds.init_dataset()
while True:
    try:
        train_batch_state, train_batch_policy, train_batch_policy2, train_batch_value = ds.batch()
        for i in range(len(train_batch_state)):
            state_data = train_batch_state[i]
            policy_data = train_batch_policy[i]
            policy_data2 = train_batch_policy2[i]
            value_data = train_batch_value[i]
            state_data = np.transpose(state_data, [2, 0, 1])
            my_color = 'b' if state_data[16][0][0] == 1 else 'r'
            opponent_color = 'r' if my_color == 'b' else 'b'
            color_matrix = {'b': np.array([[[1] * 9 for _ in range(10)]]),
                            'r': np.array([[[0] * 9 for _ in range(10)]])}
            num_history = 8
            for j in range(num_history):
                # if j < 7:
                #     continue
                if j % 2 == 0:
                    color = color_matrix[opponent_color]
                else:
                    color = color_matrix[my_color]
                if (state_data[j] == 0).all():
                    print("empty!!!")
                    continue
                current_state = np.append(state_data[j:j + 1], state_data[j + num_history:j + num_history + 1],
                                          axis=0) * 7
                current_state = np.append(current_state, color, axis=0)
                # print(current_state)
                print("start!")

                # env.print_env(state=current_state)
            env.print_env(state=current_state)
            legal_actions = env.get_all_actions(current_state)
            if legal_actions:
                legal_action_probs = filter_action_probs(policy_data, policy_data2, legal_actions, env)
                actions = list(zip(legal_actions, legal_action_probs))
                actions = sorted(actions, key=lambda x: x[1], reverse=True)
                # for action in actions:
                #     print(action)
            else:
                print("no actions to do")
            print("value", value_data, my_color)
        # break
    except tf.errors.OutOfRangeError:
        break
