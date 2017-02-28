from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from game.game import Game
import numpy as np
import time
import json
from env import korean_chess_util as kcu
from env.korean_chess import KoreanChess
import os
import shutil

# you can restore state_list
state_list_file = None  # open('./state_list.json') if os.path.isfile('./state_list.json') else None
restore_state_list = json.load(state_list_file) if state_list_file else None
Game.register('KoreanChess',
              {'position_type': 'random', 'state_list': restore_state_list if restore_state_list else None,
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

if os.path.isfile('./q_state_link.txt'):
    q_state_link = open('./q_state_link.txt')
    state_links = {}
    for line in q_state_link:
        link = json.loads(line.strip())
        state_links[list(link.keys())[0]] = list(link.values())[0]

    env.state_links = state_links

Q_blue = restore_q_blue
Q_red = restore_q_red

dis = .99
num_episodes = 100

blue_reward_list = []
red_reward_list = []
blue_win_cnt = 0
red_win_cnt = 0

records_path = 'records.txt'
for i in range(num_episodes):
    k = 0

    records_file = open(records_path)
    for line in records_file:
        start = time.time()
        line = line.strip()
        records = json.loads(line)
        record_count = len(records['records'])
        winner = records['winner']
        record_file_name = records['file']
        record_file_line = records['line']
        env.set_property('position_type', [records['blue_position_type'], records['red_position_type']])
        records = records['records']

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
        while j <= record_count:
            # blue action
            if j < record_count:
                blue_action = env.get_action_with_record(Q_blue, blue_state, records[j])
                if blue_action is not False:
                    red_state, blue_reward, blue_done, is_draw = env.step(blue_action, blue_state)
                    if record_count - 2 <= j and winner is kcu.BLUE:
                        blue_reward = KoreanChess.REWARD_LIST[kcu.KING]
                else:
                    break
                env.print_map(red_state, kcu.RED, str(i) + ':' + str(k), j, blue_reward_all, red_reward_all,
                              kcu.BLUE if blue_done else False,
                              is_draw, blue_win_cnt, red_win_cnt, Q1=Q_blue, Q2=Q_red, file=record_file_name,
                              line=record_file_line)

            # red update
            if old_red_state:
                env.init_q_state(Q_red, red_state, True)
                cur_reward = (red_reward - blue_reward)
                # cur_reward = red_reward
                if red_state in Q_red and len(Q_red[red_state]) > 0:
                    cur_q_value = cur_reward + dis * np.max(Q_red[red_state])
                else:
                    cur_q_value = cur_reward
                action_cnt = len(env.state_list[env.reverse_state_key(old_red_state)]['action_list'])
                if action_cnt != len(Q_red[old_red_state]):
                    Q_red[old_red_state] = np.zeros(action_cnt)
                Q_red[old_red_state][red_action] = cur_q_value
                red_reward_all += (red_reward - blue_reward)

            if j >= record_count:
                break

            # red action
            j += 1
            if j < record_count:
                red_action = env.get_action_with_record(Q_red, red_state, records[j], True)
                if red_action is not False:
                    next_blue_state, red_reward, red_done, is_draw = env.step(red_action, red_state, True)
                    if record_count - 2 <= j and winner is kcu.RED:
                        red_reward = KoreanChess.REWARD_LIST[kcu.KING]
                else:
                    break
                env.print_map(next_blue_state, kcu.BLUE, str(i) + ':' + str(k), j, blue_reward_all, red_reward_all,
                              kcu.RED if red_done else False,
                              is_draw, blue_win_cnt, red_win_cnt, Q1=Q_blue, Q2=Q_red, file=record_file_name,
                              line=record_file_line)

            # blue update
            env.init_q_state(Q_blue, next_blue_state)
            cur_reward = (blue_reward - red_reward)
            # cur_reward = blue_reward
            if next_blue_state in Q_blue and len(Q_blue[next_blue_state]) > 0:
                cur_q_value = cur_reward + dis * np.max(Q_blue[next_blue_state])
            else:
                cur_q_value = cur_reward
            action_cnt = len(env.state_list[blue_state]['action_list'])
            if action_cnt != len(Q_blue[blue_state]):
                Q_blue[blue_state] = np.zeros(action_cnt)
            Q_blue[blue_state][blue_action] = cur_q_value
            blue_reward_all += (blue_reward - red_reward)

            # ready for next
            blue_state = next_blue_state
            old_red_state = red_state
            j += 1

        blue_reward_list.append(blue_reward_all)
        red_reward_list.append(red_reward_all)
        # time.sleep(1)
        k += 1
        print(time.time() - start)

    records_file.close()

    # if i % 10 is 0 and i is not 0:
    env.state_list = {}
    env.state_links = {}

if os.path.isfile('./q_blue_with_data.txt'):
    shutil.move('./q_blue_with_data.txt', './q_blue_with_data_bak.txt')

with open('./q_blue_with_data.txt', 'w') as outfile:
    for key in Q_blue:
        outfile.write(key + "\n" + json.dumps(Q_blue[key].tolist()) + "\n")
    outfile.close()

if os.path.isfile('./q_red_with_data.txt'):
    shutil.move('./q_red_with_data.txt', './q_red_with_data_bak.txt')
with open('./q_red_with_data.txt', 'w') as outfile:
    for key in Q_red:
        outfile.write(key + "\n" + json.dumps(Q_red[key].tolist()) + "\n")
    outfile.close()

if os.path.isfile('./q_state_link.txt'):
    shutil.move('./q_state_link.txt', './q_state_link_bak.txt')
with open('./q_state_link.txt', 'w') as outfile:
    for key in env.state_links:
        outfile.write(json.dumps({key: env.state_links[key]}) + "\n")
    outfile.close()
