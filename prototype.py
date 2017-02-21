# coding=utf8

import numpy as np
import random
import copy
import sys
import os
import time


class ActionSpace(object):
    n = None

    def __init__(self, n):
        self.n = n


class KoreanJanggiActionSpace(ActionSpace):
    default_action_space_n = 40

    def __init__(self):
        ActionSpace.__init__(self, KoreanJanggiActionSpace.default_action_space_n)


class Environment(object):
    properties = None
    action_space = None

    def __init__(self, properties):
        self.properties = properties
        self.action_space = KoreanJanggiActionSpace()

    def reset(self):
        return 2

    def step(self, action):
        return 2


class KoreanJanggi(Environment):
    KING = 46
    SOLDIER = 1
    SANG = 2
    GUARDIUN = 3
    HORSE = 4
    CANNON = 5
    CAR = 6
    PIECE_LIST = {'r1': '졸(홍)', 'r2': '상(홍)', 'r3': '사(홍)', 'r4': '마(홍)', 'r5': '포(홍)', 'r6': '차(홍)', 'r46': '궁(홍)',
                  'b1': '졸(청)', 'b2': '상(청)', 'b3': '사(청)', 'b4': '마(청)', 'b5': '포(청)', 'b6': '차(청)', 'b46': '궁(청)', 0: '-----'}

    state_list = {}
    rand_position_list = ['masangmasang', 'masangsangma', 'sangmasangma', 'sangmamasang']
    default_state_map = [
        ['r6', 0, 0, 'r3', 0, 'r3', 0, 0, 'r6'],
        [0, 0, 0, 0, 'r46', 0, 0, 0, 0],
        [0, 'r5', 0, 0, 0, 0, 0, 'r5', 0],
        ['r1', 0, 'r1', 0, 'r1', 0, 'r1', 0, 'r1'],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ['b1', 0, 'b1', 0, 'b1', 0, 'b1', 0, 'b1'],
        [0, 'b5', 0, 0, 0, 0, 0, 'b5', 0],
        [0, 0, 0, 0, 'b46', 0, 0, 0, 0],
        ['b6', 0, 0, 'b3', 0, 'b3', 0, 0, 'b6'],
    ]

    def __init__(self, properties, state_list={}):
        Environment.__init__(self, properties)
        self.state_list = state_list

    @staticmethod
    def convert_state_key(state_map, turn):
        return str(state_map).replace('[', '').replace(', ', ',').replace(']', '').replace("'", '') + '_' + turn

    @staticmethod
    def get_available_actions(state_map):
        return [0] * 50

    def reset(self):
        if not self.properties['position_type'] or self.properties['position_type'] == 'random':
            before_rand_position = random.randint(0, 3)
            after_rand_position = random.randint(0, 3)
            position_type_list = [KoreanJanggi.rand_position_list[before_rand_position],
                                  KoreanJanggi.rand_position_list[after_rand_position]]
        else:
            position_type_list = self.properties['position_type']

        default_map = copy.deepcopy(KoreanJanggi.default_state_map)

        for i, position_type in enumerate(position_type_list):
            if position_type == 'masangmasang':
                if i == 0:
                    default_map[-1][1] = 'b4'
                    default_map[-1][2] = 'b2'
                    default_map[-1][6] = 'b4'
                    default_map[-1][7] = 'b2'
                else:
                    default_map[0][1] = 'r2'
                    default_map[0][2] = 'r4'
                    default_map[0][6] = 'r2'
                    default_map[0][7] = 'r4'
            elif position_type == 'masangsangma':
                if i == 0:
                    default_map[-1][1] = 'b4'
                    default_map[-1][2] = 'b2'
                    default_map[-1][6] = 'b2'
                    default_map[-1][7] = 'b4'
                else:
                    default_map[0][1] = 'r4'
                    default_map[0][2] = 'r2'
                    default_map[0][6] = 'r2'
                    default_map[0][7] = 'r4'
            elif position_type == 'sangmasangma':
                if i == 0:
                    default_map[-1][1] = 'b2'
                    default_map[-1][2] = 'b4'
                    default_map[-1][6] = 'b2'
                    default_map[-1][7] = 'b4'
                else:
                    default_map[0][1] = 'r4'
                    default_map[0][2] = 'r2'
                    default_map[0][6] = 'r4'
                    default_map[0][7] = 'r2'
            elif position_type == 'sangmamasang':
                if i == 0:
                    default_map[-1][1] = 'b2'
                    default_map[-1][2] = 'b4'
                    default_map[-1][6] = 'b4'
                    default_map[-1][7] = 'b2'
                else:
                    default_map[0][1] = 'r2'
                    default_map[0][2] = 'r4'
                    default_map[0][6] = 'r4'
                    default_map[0][7] = 'r2'
            else:
                raise Exception('position_type is invalid : ' + position_type)
        state_key = KoreanJanggi.convert_state_key(default_map, 'b')
        if state_key not in self.state_list:
            self.state_list[state_key] = \
                {'state_map': default_map, 'action_list': KoreanJanggi.get_available_actions(default_map), 'turn': 'b'}

        self.print_map(state_key)

        return state_key

    def print_map(self, state):
        time.sleep(1)
        os.system('clear')
        for line in self.state_list[state]['state_map']:
            converted_line = [KoreanJanggi.PIECE_LIST[val] for val in line]
            # sys.stdout.write('\r' + ' '.join(converted_line))
            print(' '.join(converted_line))
        # print('======================================================')
        # sys.stdout.flush()


    def step(self, action, state):
        # action

        # create new_state and append it
        #  to state_list, if new_state is not in state_list.

        # print next state
        self.print_map(state)
        sys.exit()

        # return new_state, reward, is_done
        return 2

    def get_action(self, Q, state, i, is_red=False):
        if is_red:
            # reverse state
            state
        if not Q or state not in Q:
            # if state is not in the Q, create state map and actions by state hash key
            Q[state] = np.zeros([len(self.state_list[state]['action_list'])])
        action_cnt = len(Q[state])
        return np.argmax(Q[state] + np.random.randn(1, action_cnt) / (i + 1))


class IrelGym(object):
    env_class_map = {'KoreanJanggi': KoreanJanggi}

    properties = {}

    @staticmethod
    def register(id, properties):
        IrelGym.properties[id] = properties

    @staticmethod
    def make(game_id):
        if not game_id:
            raise Exception('game id is not exist.')
        if game_id not in IrelGym.env_class_map:
            raise Exception('Unknown game_id.')
        return IrelGym.env_class_map[game_id](IrelGym.properties[game_id])

    def reset(self):
        return 2


def get_action(Q, state, action_space_cnt):
    return np.argmax(Q[state, :] + np.random.randn(1, action_space_cnt) / (i + 1))


IrelGym.register('KoreanJanggi', {'position_type': 'random'})

env = IrelGym.make('KoreanJanggi')
# load q table if existed.
Q_blue = {}
Q_red = {}

dis = .99
num_episodes = 2000

blue_reward_list = []
red_reward_list = []

for i in range(num_episodes):
    blue_state = env.reset()
    blue_reward_all = 0
    red_reward_all = 0
    blue_done = False
    red_done = False

    while not blue_done and not red_done:
        blue_action = env.get_action(Q_blue, blue_state, i)
        red_state, blue_reward, blue_done = env.step(blue_action, blue_state)

        if old_red_state:
            Q_red[old_red_state][red_action] = (red_reward - blue_reward) + dis * np.max(Q_red[red_state])
            red_reward_all += (red_reward - blue_reward)

        red_action = env.get_action(Q_red, red_state, i, True)
        next_blue_state, red_reward, red_done, _ = env.step(red_action, red_state)

        Q_blue[blue_state][blue_action] = (blue_reward - red_reward) + dis * np.max(Q_blue[next_blue_state])
        blue_reward_all += (blue_reward - red_reward)

        blue_state = next_blue_state
        old_red_state = red_state

    blue_reward_list.append(blue_reward_all)
    red_reward_list.append(red_reward_all)
