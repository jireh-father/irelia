# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from environment.environment import Environment
import numpy as np
import random
import copy
import sys
import os
import time

BLUE = 'b'
RED = 'r'

FORWARD = 0
RIGHT = 1
LEFT = 2
BACKWARD = 3
DIAGONAL_FORWARD_RIGHT1 = 4
DIAGONAL_FORWARD_RIGHT2 = 5
DIAGONAL_FORWARD_LEFT1 = 6
DIAGONAL_FORWARD_LEFT2 = 7
DIAGONAL_BACKWARD_RIGHT1 = 8
DIAGONAL_BACKWARD_RIGHT2 = 9
DIAGONAL_BACKWARD_LEFT1 = 10
DIAGONAL_BACKWARD_LEFT2 = 11

class KoreanChess(Environment):
    KING = 46
    SOLDIER = 1
    SANG = 2
    GUARDIAN = 3
    HORSE = 4
    CANNON = 5
    CAR = 6
    PIECE_LIST = {'r1': '졸(홍)', 'r2': '상(홍)', 'r3': '사(홍)', 'r4': '마(홍)', 'r5': '포(홍)', 'r6': '차(홍)', 'r46': '궁(홍)',
                  'b1': '졸(청)', 'b2': '상(청)', 'b3': '사(청)', 'b4': '마(청)', 'b5': '포(청)', 'b6': '차(청)', 'b46': '궁(청)',
                  0: '------'}

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
    def convert_state_key(state_map, side):
        return str(state_map).replace('[', '').replace(', ', ',').replace(']', '').replace("'", '') + '_' + side

    @staticmethod
    def get_available_actions(state_map, side):
        action_list = []
        for y, line in enumerate(state_map):
            for x, piece in enumerate(line):
                if piece == 0 or piece[0] != side:
                    continue
                action_list += KoreanChess.get_piece_actions(state_map, x, y)

        return action_list

    @staticmethod
    def get_piece_actions(state_map, x, y):
        piece_num = int(state_map[y][x][1:])
        if piece_num == KoreanChess.KING:
            return KoreanChess.get_king_actions(state_map, x, y)
        elif piece_num == KoreanChess.SOLDIER:
            return KoreanChess.get_soldier_actions(state_map, x, y)
        elif piece_num == KoreanChess.SANG:
            return KoreanChess.get_sang_actions(state_map, x, y)
        elif piece_num == KoreanChess.GUARDIAN:
            return KoreanChess.get_guardian_actions(state_map, x, y)
        elif piece_num == KoreanChess.HORSE:
            return KoreanChess.get_horse_actions(state_map, x, y)
        elif piece_num == KoreanChess.CANNON:
            return KoreanChess.get_cannon_actions(state_map, x, y)
        elif piece_num == KoreanChess.CAR:
            return KoreanChess.get_car_actions(state_map, x, y)

    @staticmethod
    def is_my_piece(state_map, x, y, side):
        return state_map[y][x] != 0 and state_map[y][x][0] == side

    @staticmethod
    def will_be_lose(state_map, x, y, to_x, to_y, side):
        return False

    @staticmethod
    def get_soldier_actions(state_map, x, y):
        piece = state_map[y][x]
        side = piece[0]

        action_list = []

        # 대각선 오른쪽 길 체크
        if ((y == 1 and x == 4) or (y == 2 and x == 3)) and \
          not KoreanChess.is_my_piece(state_map, x + 1, y - 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x + 1, y - 1, side):
            action_list.append({'x': x, 'y': y, 'way': DIAGONAL_FORWARD_RIGHT1, 'step': 1})

        # 대각선 왼쪽 길 체크
        if ((y == 1 and x == 4) or (y == 2 and x == 5)) and \
          not KoreanChess.is_my_piece(state_map, x - 1, y - 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x - 1, y - 1, side):
            action_list.append({'x': x, 'y': y, 'way': DIAGONAL_FORWARD_LEFT1, 'step': 1})

        # 전진 길 체크
        if y != 0 and not KoreanChess.is_my_piece(state_map, x, y - 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x, y - 1, side):
            action_list.append({'x': x, 'y': y, 'way': FORWARD, 'step': 1})

        # 오른쪽 길 체크
        if x != 8 and not KoreanChess.is_my_piece(state_map, x + 1, y, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x + 1, y, side):
            action_list.append({'x': x, 'y': y, 'way': RIGHT, 'step': 1})

        # 왼쪽 길 체크
        if x != 0 and not KoreanChess.is_my_piece(state_map, x - 1, y, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x - 1, y, side):
            action_list.append({'x': x, 'y': y, 'way': LEFT, 'step': 1})

        return action_list

    @staticmethod
    def get_sang_actions(state_map, x, y):
        piece = state_map[y][x]

        side = piece[0]

        action_list = []

        # 대각선 오른쪽 전진 1 길 체크


        # 대각선 오른쪽 전진 2 길 체크

        # 대각선 왼쪽 전진 1 길 체크

        # 대각선 왼쪽 전진 2 길 체크

        # 대각선 오른쪽 후진 1 길 체크

        # 대각선 오른쪽 후진 2 길 체크

        # 대각선 왼쪽 후진 1 길 체크

        # 대각선 왼쪽 후진 2 길 체크

        # [공통]
        # 우리편있는지 체크
        # 장군 검증

        return action_list

    @staticmethod
    def get_guardian_actions(state_map, x, y):
        piece = state_map[y][x]

        side = piece[0]

        action_list = []

        # 대각선 오른쪽 전진 길 체크
        if ((y == 9 and x == 3) or (y == 8 and x == 4)) and \
          not KoreanChess.is_my_piece(state_map, x + 1, y - 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x + 1, y - 1, side):
            action_list.append({'x': x, 'y': y, 'way': DIAGONAL_FORWARD_RIGHT1, 'step': 1})

        # 대각선 왼쪽 전진 길 체크
        if ((y == 9 and x == 5) or (y == 8 and x == 4)) and \
          not KoreanChess.is_my_piece(state_map, x - 1, y - 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x - 1, y - 1, side):
            action_list.append({'x': x, 'y': y, 'way': DIAGONAL_FORWARD_LEFT1, 'step': 1})

        # 대각선 오른쪽 후진 길 체크
        if ((y == 7 and x == 3) or (y == 8 and x == 4)) and \
          not KoreanChess.is_my_piece(state_map, x + 1, y + 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x + 1, y + 1, side):
            action_list.append({'x': x, 'y': y, 'way': DIAGONAL_BACKWARD_RIGHT1, 'step': 1})

        # 대각선 왼쪽 후진 길 체크
        if ((y == 7 and x == 3) or (y == 8 and x == 4)) and \
          not KoreanChess.is_my_piece(state_map, x - 1, y + 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x - 1, y + 1, side):
            action_list.append({'x': x, 'y': y, 'way': DIAGONAL_BACKWARD_RIGHT1, 'step': 1})

        # 전진 길 체크
        if y in (8, 9) and not KoreanChess.is_my_piece(state_map, x, y - 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x, y - 1, side):
            action_list.append({'x': x, 'y': y, 'way': FORWARD, 'step': 1})

        # 오른쪽 길 체크
        if x in (3, 4) and not KoreanChess.is_my_piece(state_map, x + 1, y, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x + 1, y, side):
            action_list.append({'x': x, 'y': y, 'way': RIGHT, 'step': 1})

        # 왼쪽 길 체크
        if x in (4, 5) and not KoreanChess.is_my_piece(state_map, x - 1, y, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x - 1, y, side):
            action_list.append({'x': x, 'y': y, 'way': LEFT, 'step': 1})

        # 후진 길 체크
        if y in (7, 8) and not KoreanChess.is_my_piece(state_map, x, y + 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x, y + 1, side):
            action_list.append({'x': x, 'y': y, 'way': BACKWARD, 'step': 1})

        return action_list

    @staticmethod
    def get_horse_actions(state_map, x, y):
        # 대각선 오른쪽 전진 1 길 체크

        # 대각선 오른쪽 전진 2 길 체크

        # 대각선 왼쪽 전진 1 길 체크

        # 대각선 왼쪽 전진 2 길 체크

        # 대각선 오른쪽 후진 1 길 체크

        # 대각선 오른쪽 후진 2 길 체크

        # 대각선 왼쪽 후진 1 길 체크

        # 대각선 왼쪽 후진 2 길 체크

        # [공통]
        # 우리편있는지 체크
        # 장군 검증

        return []

    @staticmethod
    def get_cannon_actions(state_map, x, y):
        piece = state_map[y][x]

        side = piece[0]

        action_list = []

        # 대각선 오른쪽 전진 길 체크

        # 대각선 왼쪽 전진 길 체크

        # 대각선 오른쪽 후진 길 체크

        # 대각선 왼쪽 후진 길 체크

        # 전진 길 체크

        # 오른쪽 길 체크

        # 왼쪽 길 체크

        # 후진 길 체크

        return action_list

    @staticmethod
    def get_car_actions(state_map, x, y):
        piece = state_map[y][x]

        side = piece[0]

        action_list = []

        # 대각선 오른쪽 전진 길 체크
        if (y == 9 and x == 3) or (y == 2 and x == 3):
            moving_y = y - 1
            moving_x = x + 1
            step = 1
            while moving_y is not -1:
                if not KoreanChess.is_my_piece(state_map, x, moving_y, side) and \
                  not KoreanChess.will_be_lose(state_map, x, y, x, moving_y, side):
                    action_list.append({'x': x, 'y': y, 'way': DIAGONAL_FORWARD_RIGHT1, 'step': step})
                moving_y = y - 1
                step += 1
        if (y == 8 and x == 4) or (y == 1 and x == 4):
            1

        # 대각선 왼쪽 전진 길 체크
        if ((y == 9 and x == 5) or (y == 8 and x == 4)) and \
          not KoreanChess.is_my_piece(state_map, x - 1, y - 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x - 1, y - 1, side):
            action_list.append({'x': x, 'y': y, 'way': DIAGONAL_FORWARD_LEFT1, 'step': 1})

        # 대각선 오른쪽 후진 길 체크
        if ((y == 7 and x == 3) or (y == 8 and x == 4)) and \
          not KoreanChess.is_my_piece(state_map, x + 1, y + 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x + 1, y + 1, side):
            action_list.append({'x': x, 'y': y, 'way': DIAGONAL_BACKWARD_RIGHT1, 'step': 1})

        # 대각선 왼쪽 후진 길 체크
        if ((y == 7 and x == 3) or (y == 8 and x == 4)) and \
          not KoreanChess.is_my_piece(state_map, x + 1, y + 1, side) and \
          not KoreanChess.will_be_lose(state_map, x, y, x + 1, y + 1, side):
            action_list.append({'x': x, 'y': y, 'way': DIAGONAL_BACKWARD_RIGHT1, 'step': 1})

        # 전진 길 체크
        if y is not 0:
            moving_y = y - 1
            step = 1
            while moving_y is not -1:
                if not KoreanChess.is_my_piece(state_map, x, moving_y, side) and \
                  not KoreanChess.will_be_lose(state_map, x, y, x, moving_y, side):
                    action_list.append({'x': x, 'y': y, 'way': FORWARD, 'step': step})
                moving_y = y - 1
                step += 1

        # 오른쪽 길 체크
        if x is not 8:
            moving_x = x + 1
            step = 1
            while moving_x is not 9:
                if not KoreanChess.is_my_piece(state_map, moving_x, y, side) and \
                  not KoreanChess.will_be_lose(state_map, x, y, moving_x, y, side):
                    action_list.append({'x': x, 'y': y, 'way': RIGHT, 'step': step})
                moving_x = x + 1
                step += 1

        # 왼쪽 길 체크
        if x is not 0:
            moving_x = x - 1
            step = 1
            while moving_x is not -1:
                if not KoreanChess.is_my_piece(state_map, moving_x, y, side) and \
                  not KoreanChess.will_be_lose(state_map, x, y, moving_x, y, side):
                    action_list.append({'x': x, 'y': y, 'way': LEFT, 'step': step})
                moving_x = x - 1
                step += 1

        # 후진 길 체크
        if y < 9:
            moving_y = y + 1
            step = 1
            while moving_y < 10:
                if not KoreanChess.is_my_piece(state_map, x, moving_y, side) and \
                  not KoreanChess.will_be_lose(state_map, x, y, x, moving_y, side):
                    action_list.append({'x': x, 'y': y, 'way': BACKWARD, 'step': step})
                moving_y = y + 1
                step += 1

        return action_list

    def reset(self):
        if not self.properties['position_type'] or self.properties['position_type'] == 'random':
            before_rand_position = random.randint(0, 3)
            after_rand_position = random.randint(0, 3)
            position_type_list = [KoreanChess.rand_position_list[before_rand_position],
                                  KoreanChess.rand_position_list[after_rand_position]]
        else:
            position_type_list = self.properties['position_type']

        default_map = copy.deepcopy(KoreanChess.default_state_map)

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
        state_key = KoreanChess.convert_state_key(default_map, 'b')
        if state_key not in self.state_list:
            self.state_list[state_key] = \
                {'state_map': default_map, 'action_list': KoreanChess.get_available_actions(default_map, 'b'),
                 'side': 'b'}

        self.print_map(state_key)

        return state_key

    def print_map(self, state):
        time.sleep(1)
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
        # sys.stdout.flush()
        for line in self.state_list[state]['state_map']:
            converted_line = [KoreanChess.PIECE_LIST[val] for val in line]
            # sys.stdout.write('\r' + ' '.join(converted_line))
            print(' '.join(converted_line))
            # print('======================================================')

    def step(self, action, state):
        # action

        # create new_state and append it
        #  to state_list, if new_state is not in state_list.

        # 상태 새로 만들었는데 action이 없으면 끝

        # print next state
        self.print_map(state)
        sys.exit()

        # return new_state, reward, is_done
        # 다음 상태에 액션이 없으면 is_done
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
