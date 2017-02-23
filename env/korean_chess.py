# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import random
import sys
import time
import numpy as np

from env.env import Env
from env.korean_chess_piece import piece_factory
from env import korean_chess_util as kcu


class KoreanChess(Env):
    PIECE_LIST = {'r1': '졸(홍)', 'r2': '상(홍)', 'r3': '사(홍)', 'r4': '마(홍)', 'r5': '포(홍)', 'r6': '차(홍)', 'r7': '궁(홍)',
                  'b1': '졸(청)', 'b2': '상(청)', 'b3': '사(청)', 'b4': '마(청)', 'b5': '포(청)', 'b6': '차(청)', 'b7': '궁(청)',
                  0: '------'}

    state_list = {}
    rand_position_list = ['masangmasang', 'masangsangma', 'sangmasangma', 'sangmamasang']
    default_state_map = [
        ['r6', 0, 0, 'r3', 0, 'r3', 0, 0, 'r6'],
        [0, 0, 0, 0, 'r7', 0, 0, 0, 0],
        [0, 'r5', 0, 0, 0, 0, 0, 'r5', 0],
        ['r1', 0, 'r1', 0, 'r1', 0, 'r1', 0, 'r1'],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ['b1', 0, 'b1', 0, 'b1', 0, 'b1', 0, 'b1'],
        [0, 'b5', 0, 0, 0, 0, 0, 'b5', 0],
        [0, 0, 0, 0, 'b7', 0, 0, 0, 0],
        ['b6', 0, 0, 'b3', 0, 'b3', 0, 0, 'b6'],
    ]

    POSITION_TYPE_LIST = {
        'masangmasang': [
            ['b6', 'b4', 'b2', 'b3', 0, 'b3', 'b4', 'b2', 'b6'],
            ['r6', 'r2', 'r4', 'r3', 0, 'r3', 'r2', 'r4', 'r6'],
        ],
        'masangsangma': [
            ['b6', 'b4', 'b2', 'b3', 0, 'b3', 'b2', 'b4', 'b6'],
            ['r6', 'r4', 'r2', 'r3', 0, 'r3', 'r2', 'r4', 'r6'],
        ],
        'sangmasangma': [
            ['b6', 'b2', 'b4', 'b3', 0, 'b3', 'b2', 'b4', 'b6'],
            ['r6', 'r4', 'r2', 'r3', 0, 'r3', 'r4', 'r2', 'r6'],
        ],
        'sangmamasang': [
            ['b6', 'b2', 'b4', 'b3', 0, 'b3', 'b4', 'b2', 'b6'],
            ['r6', 'r2', 'r4', 'r3', 0, 'r3', 'r4', 'r2', 'r6'],
        ],
    }

    REWARD_LIST = {
        kcu.SOLDIER: 1,
        kcu.SANG: 2,
        kcu.GUARDIAN: 3,
        kcu.HORSE: 4,
        kcu.CANNON: 5,
        kcu.CAR: 6,
        kcu.KING: 46,
    }

    def __init__(self, properties):
        Env.__init__(self, properties)
        if self.properties and 'state_list' in self.properties:
            self.state_list = self.properties['state_list']
        if not self.state_list:
            self.state_list = {}

    @staticmethod
    def convert_state_key(state_map):
        return str(state_map).replace('[', '').replace(', ', ',').replace(']', '').replace("'", '')

    @staticmethod
    def get_actions(state_map, side):
        action_list = []
        for y, line in enumerate(state_map):
            for x, piece in enumerate(line):
                if piece == 0 or piece[0] != side:
                    continue
                action_list += KoreanChess.get_piece_actions(state_map, x, y)

        return action_list

    @staticmethod
    def get_piece_actions(state_map, x, y):
        return piece_factory.get_actions(state_map, x, y)

    def action(self, state_key, action_key, is_red=False):
        if is_red:
            state = self.state_list[self.reverse_state_key(state_key)]
        else:
            state = self.state_list[state_key]

        return piece_factory.action(state['state_map'], state['action_list'][action_key])

    def reset(self):
        if self.properties['init_state']:
            default_map = self.properties['init_state']
            side = self.properties['init_side'] if self.properties['init_side'] else 'b'
        else:
            side = kcu.BLUE
            if not self.properties['position_type'] or self.properties['position_type'] == 'random':
                before_rand_position = random.randint(0, 3)
                after_rand_position = random.randint(0, 3)
                position_type_list = [KoreanChess.rand_position_list[before_rand_position],
                                      KoreanChess.rand_position_list[after_rand_position]]
            else:
                position_type_list = self.properties['position_type']

            default_map = copy.deepcopy(KoreanChess.default_state_map)

            for i, position_type in enumerate(position_type_list):
                if position_type not in KoreanChess.POSITION_TYPE_LIST:
                    raise Exception('position_type is invalid : ' + position_type)

                line_idx = -1 if i == 0 else 0

                default_map[line_idx] = KoreanChess.POSITION_TYPE_LIST[position_type][i]

        state_key = self.create_state(default_map, side)

        self.print_map(state_key, side)

        for action in self.state_list[state_key]['action_list']:
            print(action)

        return state_key

    def create_state(self, state_map, side):
        state_key = KoreanChess.convert_state_key(state_map)

        if state_key not in self.state_list:
            self.state_list[state_key] = {'state_map': state_map,
                                          'action_list': KoreanChess.get_actions(state_map, side), 'side': side}

        if side is kcu.RED:
            return self.reverse_state_key(state_key)
        else:
            return state_key

    def print_map(self, state, side):
        time.sleep(0.5)
        # if os.name == 'nt':
        #     os.system('cls')
        # else:
        #     os.system('clear')
        # sys.stdout.flush()
        if side is kcu.RED:
            state = self.reverse_state_key(state)
        for line in self.state_list[state]['state_map']:
            converted_line = [KoreanChess.PIECE_LIST[val] for val in line]
            # sys.stdout.write('\r' + ' '.join(converted_line))
            print(' '.join(converted_line))
            # print('======================================================')

    def step(self, action, state_key, is_red=False):
        opposite_side = kcu.BLUE if is_red else kcu.RED
        # action
        # new_state_map 은 현재 state_map 대비 뒤집어진 상태로 나온다.
        new_state_map, reward, is_done = self.action(state_key, action, is_red)

        # create new_state and append it
        #  to state_list, if new_state is not in state_list.
        new_state_key = self.create_state(new_state_map, opposite_side)

        # print next state
        self.print_map(new_state_key, opposite_side)

        # return new_state, reward, is_done
        return new_state_key, reward, is_done

    def reverse_state_key(self, state):
        return self.convert_state_key(list(reversed(state.split(','))))

    def get_action(self, Q, state, i, is_red=False):
        if not Q or state not in Q:
            # if state is not in the Q, create state map and actions by state hash key
            if is_red:
                # reverse state
                action_cnt = len(self.state_list[self.reverse_state_key(state)]['action_list'])
            else:
                action_cnt = len(self.state_list[state]['action_list'])
            if action_cnt > 0:
                Q[state] = np.zeros(action_cnt)

        action_cnt = len(Q[state])
        if action_cnt < 1:
            return False
        else:
            return np.argmax(Q[state] + np.random.randn(1, action_cnt) / (i + 1))
