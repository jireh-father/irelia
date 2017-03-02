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

# you can restore state_list
state_list_file = None  # open('./state_list.json') if os.path.isfile('./state_list.json') else None
restore_state_list = json.load(state_list_file) if state_list_file else None
Game.register('KoreanChess',
              {'position_type': 'random',
               'state_list': restore_state_list if restore_state_list else None,
               'init_state': None, 'init_side': 'b'})

env = Game.make('KoreanChess')

# load q table if existed.
restore_q_blue = {}
restore_q_red = {}
if os.path.isfile('./q_blue_with_data.txt'):
    q_file = open('./q_blue_with_data.txt')
    i = 0
    key = None
    for line in q_file:
        if i % 2 is 0:
            key = line.strip()
        else:
            restore_q_blue[key] = np.array(json.loads(line.strip()))
        i += 1

if os.path.isfile('./q_red_with_data.txt'):
    q_file = open('./q_red_with_data.txt')
    i = 0
    key = None
    for line in q_file:
        if i % 2 is 0:
            key = line.strip()
        else:
            restore_q_red[key] = np.array(json.loads(line.strip()))
        i += 1

Q_blue = restore_q_blue
Q_red = restore_q_red

blue_reward_list = []
red_reward_list = []
blue_win_cnt = 0
red_win_cnt = 0

blue_state = env.reset()
init_map = env.state_list[blue_state]['state_map']
# history.append(blue_state)
env.print_map_for_test(blue_state, kcu.BLUE)
blue_reward_all = 0
red_reward_all = 0
blue_done = False
red_done = False
old_red_state = None
red_action = False
is_draw = False
j = 0
while not blue_done and not red_done:
    blue_action = env.get_action_test(Q_blue, blue_state)
    red_state, blue_reward, blue_done, is_draw = env.step(blue_action, blue_state)
    # history.append(red_state)
    env.print_map_for_test(red_state, kcu.RED, 0, j, blue_reward_all, red_reward_all,
                           kcu.BLUE if blue_done else False,
                           is_draw, blue_win_cnt, red_win_cnt, Q1=Q_blue, Q2=Q_red)
    if blue_action is False and red_action is False:
        break

    if old_red_state:
        env.init_q_state(Q_red, red_state, True)

    if blue_done and not is_draw:
        blue_win_cnt += 1
        break

    red_action = env.get_action_test(Q_red, red_state, True)
    next_blue_state, red_reward, red_done, is_draw = env.step(red_action, red_state, True)
    # history.append(next_blue_state)
    env.print_map_for_test(next_blue_state, kcu.BLUE, 0, j, blue_reward_all, red_reward_all,
                           kcu.RED if red_done else False,
                           is_draw, blue_win_cnt, red_win_cnt, Q1=Q_blue, Q2=Q_red)

    if blue_action is False and red_action is False:
        break

    env.init_q_state(Q_blue, next_blue_state)

    if red_done and not is_draw:
        red_win_cnt += 1
        break

    blue_state = next_blue_state
    old_red_state = red_state
    j += 1

    if os.path.isfile('./test_history.txt'):
        shutil.move('./test_history.txt', './test_history_bak.txt')
    with open('./test_history.txt', 'w') as outfile:
        outfile.write(json.dumps({'map': init_map, 'review_history': env.history}))
        outfile.close()

blue_reward_list.append(blue_reward_all)
red_reward_list.append(red_reward_all)
