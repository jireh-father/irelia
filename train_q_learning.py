from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from game.game import Game
import numpy as np
import time
import json
from env import korean_chess_util as kcu
import os

# you can restore state_list
state_list_file = None#open('./state_list.json') if os.path.isfile('./state_list.json') else None
restore_state_list = json.load(state_list_file) if state_list_file else None
Game.register('KoreanChess',
              {'position_type': 'random', 'state_list': restore_state_list if restore_state_list else None,
               'init_state': None, 'init_side': 'b'})

env = Game.make('KoreanChess')

# load q table if existed.
q_blue_file = None#open('./q_blue.json') if os.path.isfile('./q_blue.json') else None
restore_q_blue = json.load(q_blue_file) if q_blue_file else None
q_red_file = None#open('./q_red.json') if os.path.isfile('./q_red.json') else None
restore_q_red = json.load(q_red_file) if q_red_file else None

Q_blue = restore_q_blue if restore_q_blue else {}
Q_red = restore_q_red if restore_q_red else {}

dis = .99
num_episodes = 1000000

blue_reward_list = []
red_reward_list = []
blue_win_cnt = 0
red_win_cnt = 0

for i in range(num_episodes):
    blue_state = env.reset()
    env.print_map(blue_state, kcu.BLUE)
    blue_reward_all = 0
    red_reward_all = 0
    blue_done = False
    red_done = False
    old_red_state = None
    red_action = False
    is_draw = False
    j = 0
    while not blue_done and not red_done:
        if j >= 500:
            break

        blue_action = env.get_action(Q_blue, blue_state, i)
        red_state, blue_reward, blue_done, is_draw = env.step(blue_action, blue_state)
        env.print_map(red_state, kcu.RED, i, j, blue_reward_all, red_reward_all, kcu.BLUE if blue_done else False,
                      is_draw, blue_win_cnt, red_win_cnt)
        if blue_action is False and red_action is False:
            break

        if old_red_state:
            env.init_q_state(Q_red, red_state, True)
            cur_reward = (red_reward - blue_reward)
            if red_state in Q_red and len(Q_red[red_state]) > 0:
                cur_q_value = cur_reward + dis * np.max(Q_red[red_state])
            Q_red[old_red_state][red_action] = cur_q_value
            red_reward_all += (red_reward - blue_reward)

        if blue_done and not is_draw:
            blue_win_cnt += 1
            break

        red_action = env.get_action(Q_red, red_state, i, True)
        next_blue_state, red_reward, red_done, is_draw = env.step(red_action, red_state, True)
        env.print_map(next_blue_state, kcu.BLUE, i, j, blue_reward_all, red_reward_all, kcu.RED if red_done else False,
                      is_draw, blue_win_cnt, red_win_cnt)

        if blue_action is False and red_action is False:
            break

        env.init_q_state(Q_blue, next_blue_state)
        cur_reward = (blue_reward - red_reward)
        if next_blue_state in Q_blue and len(Q_blue[next_blue_state]) > 0:
            cur_q_value = cur_reward + dis * np.max(Q_blue[next_blue_state])
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
    # time.sleep(1)

    if i % 200 is 0:
        with open('./q_blue.json', 'w') as outfile:
            q_blue_tmp = {key: Q_blue[key].tolist() for key in Q_blue}
            json.dump(q_blue_tmp, outfile)
        with open('./q_red.json', 'w') as outfile:
            q_red_tmp = {key: Q_red[key].tolist() for key in Q_red}
            json.dump(q_red_tmp, outfile)
    #     with open('./state_list.json', 'w') as outfile:
    #         json.dump(env.state_list, outfile)
