from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from game.game import Game
import numpy as np
import time
import json
from env import korean_chess_util as kcu
import os
import shutil
import operator
from env.korean_chess import KoreanChess

# you can restore state_list
state_list_file = None  # open('./state_list.json') if os.path.isfile('./state_list.json') else None
restore_state_list = json.load(state_list_file) if state_list_file else None
Game.register('KoreanChess',
              {'position_type': 'random', 'state_list': restore_state_list if restore_state_list else None,
               'init_state': None, 'init_side': 'b'})

env = Game.make('KoreanChess')

dis = .99
num_episodes = 1000000

blue_reward_list = []
red_reward_list = []
blue_win_cnt = 0
red_win_cnt = 0

for i in range(num_episodes):
    blue_state = env.reset()
    env.print_env(blue_state, kcu.BLUE)
    blue_reward_all = 0
    red_reward_all = 0
    blue_done = False
    red_done = False
    old_red_state = None
    red_action = False
    is_draw = False
    j = 0
    while not blue_done and not red_done:
        if j >= 400:
            break

        blue_action = env.get_action_es(blue_state, kcu.BLUE)
        red_state, blue_reward, blue_done, is_draw = env.step(blue_action, blue_state)
        env.print_env(red_state, kcu.RED, i, j, blue_reward_all, red_reward_all, kcu.BLUE if blue_done else False,
                      is_draw, blue_win_cnt, red_win_cnt)
        if blue_action is False and red_action is False:
            print('there is no action.')
            break

        if old_red_state:
            cur_reward = (red_reward - blue_reward * .9)
            red_q = KoreanChess.get_q_from_es(red_state, kcu.RED)

            max_action = int(max(red_q.iteritems(), key=operator.itemgetter(1))[1])

            if red_q:
                cur_q_value = cur_reward + dis * max_action
            else:
                cur_q_value = cur_reward
            KoreanChess.update_es_q(old_red_state, red_action, cur_q_value)
            red_reward_all += (red_reward - blue_reward * .9)

        if blue_done and not is_draw:
            blue_win_cnt += 1
            break

        red_action = env.get_action(Q_red, red_state, i, True)
        next_blue_state, red_reward, red_done, is_draw = env.step(red_action, red_state, True)
        env.print_env(next_blue_state, kcu.BLUE, i, j, blue_reward_all, red_reward_all, kcu.RED if red_done else False,
                      is_draw, blue_win_cnt, red_win_cnt, Q1=Q_blue, Q2=Q_red)

        if blue_action is False and red_action is False:
            break

        env.init_q_state(Q_blue, next_blue_state)
        cur_reward = (blue_reward - red_reward * .9)
        # cur_reward = blue_reward
        if next_blue_state in Q_blue and len(Q_blue[next_blue_state]) > 0:
            cur_q_value = cur_reward + dis * np.max(Q_blue[next_blue_state])
        else:
            cur_q_value = cur_reward
        Q_blue[blue_state][blue_action] = cur_q_value
        blue_reward_all += (blue_reward - red_reward)

        if red_done and not is_draw:
            red_win_cnt += 1
            break

        blue_state = next_blue_state
        old_red_state = red_state
        j += 1

    blue_reward_list.append(blue_reward_all)
    red_reward_list.append(red_reward_all)

    env.state_list = {}
    env.state_links = {}
    # time.sleep(1)


    if i % 10 is 0 and i is not 0:
        if os.path.isfile('./q_blue.txt'):
            shutil.move('./q_blue.txt', './q_blue_bak.txt')
        if os.path.isfile('./q_red.txt'):
            shutil.move('./q_red.txt', './q_red_bak.txt')
        with open('./q_blue.txt', 'w') as outfile:
            for key in Q_blue:
                outfile.write(key + "\n" + json.dumps(Q_blue[key].tolist()) + "\n")
            outfile.close()
        with open('./q_red.txt', 'w') as outfile:
            for key in Q_red:
                outfile.write(key + "\n" + json.dumps(Q_red[key].tolist()) + "\n")
            outfile.close()
