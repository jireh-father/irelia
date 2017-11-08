from game.game import Game
import traceback
from game import korean_chess_constant as c
import random
import numpy as np
import tensorflow as tf
from model import resnet
from model import actor_critic
import os
from util import user_input

LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic
N_F = 9 * 10  # env.observation_space.shape[0]
N_A = 9 * 10  # env.action_space.n

checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint", "model.ckpt")

sess = tf.Session()
ph_state = tf.placeholder(tf.float32, [1, 10, 9, 3], "state")
conv_logits = resnet.model(ph_state, blocks=20, data_format="channels_last")
conv_logits = tf.reshape(conv_logits, [-1, N_F], name="reshape")
actor = actor_critic.Actor(sess, input=conv_logits, input_ph=ph_state, n_actions=N_A, lr=LR_A)
critic = actor_critic.Critic(sess, input=conv_logits, input_ph=ph_state,
                             lr=LR_C)  # we need a good teacher, so the teacher should learn faster than the actor

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

if os.path.exists(checkpoint_path + ".index"):
    saver.restore(sess, checkpoint_path)
    
env = Game.make("KoreanChess-v1")

env.reset()
i = 0
while True:
    if i % 2 == 0:
        from_x, from_y, to_x, to_y = user_input.get_user_input()

        try:
            new_state, reward, done, _ = env.step({"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y})

            if done:
                print("The End")
                break

        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
    else:
        new_state = actor_critic.reverse_state(new_state)
        action = actor.choose_action(new_state, env)
        try:
            new_state, reward, done, _ = env.step(
                {"from_x": action["from_x"], "from_y": action["from_y"], "to_x": action["to_x"],
                 "to_y": action["to_y"]})
            if done:
                print("The End")
                break
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
    i += 1
