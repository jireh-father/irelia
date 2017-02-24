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

        action = state['action_list'][action_key]
        state_map = copy.deepcopy(state['state_map'])
        to_x = action['to_x']
        to_y = action['to_y']
        x = action['x']
        y = action['y']

        # reward 계산
        to_value = state_map[to_y][to_x]
        if to_value is 0:
            reward = 0
        else:
            reward = KoreanChess.REWARD_LIST[int(to_value[1])]

        # 이동 지점에 기존지점 말 저장
        state_map[to_y][to_x] = state_map[y][x]

        # 기존 지점 0으로 세팅
        state_map[y][x] = 0

        is_done = reward == KoreanChess.REWARD_LIST[kcu.KING] or KoreanChess.is_draw(state_map)

        # state_map 결과는 무조건 reverse해서 보내라
        return self.reverse_state_map(state_map), reward, is_done, KoreanChess.is_draw(state_map)

    def reverse_state_map(self, state_map):
        reversed_map = np.array(list(reversed(np.array(state_map).flatten()))).reshape([-1, 9]).tolist()
        result_map = []
        for line in reversed_map:
            result_map.append([int(val) if val is '0' else val for val in line])
        return result_map

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

        # self.print_map(state_key, side)

        # for action in self.state_list[state_key]['action_list']:
        #     print(action)

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

    def print_map(self, state, side, episode=0, turn=0, blue_reward_episode=0, red_reward_episode=0, done_side=False,
                  is_draw=False, blue_win_cnt=0, red_win_cnt=0, Q1=None, Q2=None):
        if turn % 20 is not 0:
            return
        time.sleep(1)
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
        # sys.stdout.flush()
        if side is kcu.RED:
            state = self.reverse_state_key(state)
            map = self.reverse_state_map(self.state_list[state]['state_map'])
        else:
            map = self.state_list[state]['state_map']
        print(
            'EPISODE {:d}, TURN {:d}, BLUE REWARD {:d}, RED REWARD {:d}'.format(episode, turn, blue_reward_episode,
                                                                                   red_reward_episode))
        if Q1 and Q2:
            print('Q1 COUNT {:d}, Q2 COUNT {:d}'.format(len(Q1), len(Q2)))
        print('TOTAL BLUE WIN {:d}, TOTAL RED WIN {:d}, TOTAL STATE COUNT {:d}'.format(blue_win_cnt, red_win_cnt,
                                                                                       len(self.state_list)))

        if is_draw:
            print('draw')
        elif done_side:
            print('WiNNER {:s}'.format(done_side))
        else:
            print('running')

        for line in map:
            converted_line = [KoreanChess.PIECE_LIST[val] for val in line]
            # sys.stdout.write('\r' + ' '.join(converted_line))
            print(' '.join(converted_line))
        #     # print('======================================================')

    def init_q_state(self, Q, state, is_red=False):
        if not Q or state not in Q:
            # if state is not in the Q, create state map and actions by state hash key
            if is_red:
                # reverse state
                action_cnt = len(self.state_list[self.reverse_state_key(state)]['action_list'])
            else:
                action_cnt = len(self.state_list[state]['action_list'])
            Q[state] = np.zeros(action_cnt)

    @staticmethod
    def convert_state_map(state_key):
        state_map = np.array(state_key.split(',')).reshape([-1, 9]).tolist()
        result_map = []
        for line in state_map:
            result_map.append([int(val) if val is '0' else val for val in line])
        return result_map

    @staticmethod
    def is_draw(state_map):
        for line in state_map:
            for piece in line:
                if piece is 0:
                    continue
                # todo: 상하고 마하고 구현해면 빼
                if piece[1] is not kcu.KING and piece[1] is not kcu.GUARDIAN and piece[1] is not kcu.HORSE and piece[
                    1] is not kcu.SANG:
                    return False
                    # todo: 포만 남았을경우 못넘는경우면 비긴걸로 계산
        return True

    def step(self, action, state_key, is_red=False):
        opposite_side = kcu.BLUE if is_red else kcu.RED

        if action is False:
            if is_red is False:
                state_key = self.reverse_state_key(state_key)
            self.create_state(self.convert_state_map(state_key), opposite_side)
            # state_list 에 새로 생성해서 추가
            return state_key, 0, False, False

        # action
        # new_state_map 은 현재 state_map 대비 뒤집어진 상태로 나온다.
        new_state_map, reward, is_done, is_draw = self.action(state_key, action, is_red)

        # create new_state and append it
        #  to state_list, if new_state is not in state_list.
        new_state_key = self.create_state(new_state_map, opposite_side)

        # print next state
        # self.print_map(new_state_key, opposite_side)

        # return new_state, reward, is_done
        return new_state_key, reward, is_done, is_draw

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
            Q[state] = np.zeros(action_cnt)

        action_cnt = len(Q[state])
        if action_cnt < 1:
            return False
        else:
            return np.argmax(Q[state] + np.random.randn(1, action_cnt) / (i + 1))
