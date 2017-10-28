# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import time
import numpy as np
import numbers

from game.korean_chess_piece import piece_factory
from game import korean_chess_util as c
import operator


class KoreanChessV1:
    PIECE_MAP_KOR = \
        {c.R_SD: '졸(홍)', c.R_SG: '상(홍)', c.R_GD: '사(홍)', c.R_HS: '마(홍)', c.R_CN: '포(홍)', c.R_CR: '차(홍)', c.R_KG: '궁(홍)',
         c.B_SD: '졸(청)', c.B_SG: '상(청)', c.B_GD: '사(청)', c.B_HS: '마(청)', c.B_CN: '포(청)', c.B_CR: '차(청)', c.B_KG: '궁(청)',
         0: '-----'}

    default_state = [
        [c.R_CR, 0, 0, c.R_GD, 0, c.R_GD, 0, 0, c.R_CR],
        [0, 0, 0, 0, c.R_KG, 0, 0, 0, 0],
        [0, c.R_CN, 0, 0, 0, 0, 0, c.R_CN, 0],
        [c.R_SD, 0, c.R_SD, 0, c.R_SD, 0, c.R_SD, 0, c.R_SD],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [c.B_SD, 0, c.B_SD, 0, c.B_SD, 0, c.B_SD, 0, c.B_SD],
        [0, c.B_CN, 0, 0, 0, 0, 0, c.B_CN, 0],
        [0, 0, 0, 0, c.B_KG, 0, 0, 0, 0],
        [c.B_CR, 0, 0, c.B_GD, 0, c.B_GD, 0, 0, c.B_CR],
    ]

    POSITION_TYPE_LIST = [
        # 마상마상
        [
            [c.B_CR, c.B_HS, c.B_SG, c.B_GD, 0, c.B_GD, c.B_HS, c.B_SG, c.B_CR],
            [c.R_CR, c.R_SG, c.R_HS, c.R_GD, 0, c.R_GD, c.R_SG, c.R_HS, c.R_CR],
        ],
        # 마상상마
        [
            [c.B_CR, c.B_HS, c.B_SG, c.B_GD, 0, c.B_GD, c.B_SG, c.B_HS, c.B_CR],
            [c.R_CR, c.R_HS, c.R_SG, c.R_GD, 0, c.R_GD, c.R_SG, c.R_HS, c.R_CR],
        ],
        # 상마상마
        [
            [c.B_CR, c.B_SG, c.B_HS, c.B_GD, 0, c.B_GD, c.B_SG, c.B_HS, c.B_CR],
            [c.R_CR, c.R_HS, c.R_SG, c.R_GD, 0, c.R_GD, c.R_HS, c.R_SG, c.R_CR],
        ],
        # 상마마상
        [
            [c.B_CR, c.B_SG, c.B_HS, c.B_GD, 0, c.B_GD, c.B_HS, c.B_SG, c.B_CR],
            [c.R_CR, c.R_SG, c.R_HS, c.R_GD, 0, c.R_GD, c.R_HS, c.R_SG, c.R_CR],
        ],
    ]

    REWARD_LIST = {
        c.SOLDIER: 2,
        c.SANG: 3,
        c.GUARDIAN: 3,
        c.HORSE: 5,
        c.CANNON: 7,
        c.CAR: 13,
        c.KING: 73,
    }

    def __init__(self, properties):
        self.properties = properties
        self.current_state = None
        self.current_turn = None

    def reset(self):
        if not self.properties or (
                "position_type" not in self.properties or self.properties['position_type'] == 'random'):
            blue_rand_position = random.randint(0, 3)
            red_rand_position = random.randint(0, 3)
            position_type_list = [blue_rand_position, red_rand_position]
        else:
            position_type_list = self.properties['position_type']

        self.current_turn = c.BLUE
        current_state = copy.deepcopy(KoreanChessV1.default_state)

        for i, position_type in enumerate(position_type_list):
            if not KoreanChessV1.POSITION_TYPE_LIST[position_type]:
                raise Exception('position_type is invalid : ' + str(position_type))

            line_idx = -1 if i == 0 else 0

            current_state[line_idx] = KoreanChessV1.POSITION_TYPE_LIST[position_type][i]
        self.current_state = current_state
        self.print_env()

        return current_state

    def step(self, action):
        if not self.validate_action(action):
            raise Exception("Invalid action :%s" % action)

        to_x = action['to_x']
        to_y = action['to_y']
        from_x = action['from_x']
        from_y = action['from_y']

        # reward
        to_value = self.current_state[to_y][to_x]
        reward = 0 if to_value is 0 else KoreanChessV1.REWARD_LIST[int(to_value[1])]

        # move
        self.current_state[to_y][to_x] = self.current_state[from_y][from_x]
        self.current_state[from_y][from_x] = 0

        is_draw = self.is_draw()
        # todo: we tong su
        # todo: time limit
        # todo: checkmate
        # todo: repeat limit
        # todo: count, win or lose by count
        # todo: turn count limit
        done = (reward == KoreanChessV1.REWARD_LIST[c.KING] or is_draw)

        self.current_turn = c.RED if self.current_turn == c.BLUE else c.BLUE

        self.print_env()

        return self.current_state, reward, done, is_draw

    def validate_action(self, action):
        to_x = action['to_x']
        to_y = action['to_y']
        from_x = action['from_x']
        from_y = action['from_y']
        piece_num = int(self.current_state[from_y][from_x][-1])
        if piece_num == 0:
            return False

        if self.current_state[from_y][from_x][0] != self.current_turn:
            return False

        if self.current_turn == c.RED:
            from_x, from_y, to_x, to_y = c.reverse_action(from_x, from_y, to_x, to_y)
            current_state = c.reverse_state(self.current_state)
        else:
            current_state = self.current_state

        piece = piece_factory.get_piece(piece_num)

        actions = piece.get_actions(current_state, from_x, from_y)
        if not actions:
            return False

        invalid_cnt = 0
        for action in actions:
            if action["to_x"] != to_x or action["to_y"] != to_y:
                invalid_cnt += 1

        return invalid_cnt != len(actions)

    @staticmethod
    def compress_state_key(state_key):
        empty_cnt = 0
        state_key_list = []
        for piece in state_key.split(','):
            if piece == '0':
                empty_cnt += 1
            else:
                if empty_cnt > 0:
                    state_key_list.append(str(empty_cnt))
                    empty_cnt = 0
                state_key_list.append(piece)
        if empty_cnt > 0:
            state_key_list.append(str(empty_cnt))

        return ','.join(state_key_list)

    @staticmethod
    def decompress_state_key(state_key):
        state_key_list = []
        for piece in state_key.split(','):
            if piece.isdigit():
                state_key_list += ['0'] * int(piece)
            else:
                state_key_list.append(piece)

        return ','.join(state_key_list)

    @staticmethod
    def convert_state_key(state_map):
        empty_cnt = 0
        state_key_list = []
        for row in state_map:
            for piece in row:
                if piece == 0:
                    empty_cnt += 1
                else:
                    if empty_cnt > 0:
                        state_key_list.append(str(empty_cnt))
                        empty_cnt = 0
                    state_key_list.append(piece)
        if empty_cnt > 0:
            state_key_list.append(str(empty_cnt))

        return ','.join(state_key_list)

    @staticmethod
    def convert_state_map(state_key):
        state_map = []
        for piece in state_key.split(','):
            if piece.isdigit():
                state_map += [0] * int(piece)
            else:
                state_map.append(piece)
        result = np.array(state_map).reshape([-1, 9]).tolist()
        for i, row in enumerate(result):
            for j, piece in enumerate(row):
                if piece == '0':
                    result[i][j] = 0
        return result

    @staticmethod
    def convert_uncompressed_state_list(state_key):
        state_map = []
        for piece in state_key.split(','):
            if piece.isdigit():
                state_map += [0] * int(piece)
            else:
                state_map.append(piece)
        return state_map

    @staticmethod
    def convert_state_list(state_key):
        state_list = KoreanChessV1.convert_uncompressed_state_list(state_key)
        converted_state = []
        for piece in state_list:
            if isinstance(piece, numbers.Integral):
                converted_state.append(int(piece))
                continue
            if piece[0] is 'r':
                converted_state.append(0 - int(piece[1:]))
            else:
                converted_state.append(int(piece[1:]))

        return converted_state

    @staticmethod
    def get_actions(state_map, side):
        action_list = []
        for y, line in enumerate(state_map):
            for x, piece in enumerate(line):
                if piece == 0 or piece[0] != side:
                    continue
                action_list += KoreanChessV1.get_piece_actions(state_map, x, y)

        return action_list


    def is_losing_way(state_map, x, y, to_x, to_y, side):

        return False

    @staticmethod
    def reverse_state_map(state_map):
        reversed_map = np.array(list(reversed(np.array(state_map).flatten()))).reshape([-1, 9]).tolist()
        result_map = []
        for line in reversed_map:
            result_map.append([int(val) if val == u'0' else val for val in line])
        return result_map

    def print_env(self, interval=0):
        if interval > 0:
            time.sleep(0.5)
        print("Y  X " + KoreanChessV1.PIECE_MAP_KOR[0].join(["%d" % col_idx for col_idx in range(0, 9)]))
        for i, line in enumerate(self.current_state):
            line = [KoreanChessV1.PIECE_MAP_KOR[piece] for piece in line]
            print("%d %s" % (i, ' '.join(line)))
            # print('======================================================')

    def init_q_state(self, Q, state, is_red=False):
        if not Q or state not in Q:
            # if state is not in the Q, create state map and actions by state hash key
            if is_red:
                # reverse state
                action_cnt = len(self.state_list[KoreanChessV1.reverse_state_key(state)]['action_list'])
            else:
                action_cnt = len(self.state_list[state]['action_list'])
            Q[state] = np.zeros(action_cnt)

    @staticmethod
    def compare_state(state_key1, state_key2):
        state_list1 = KoreanChessV1.convert_state_list(state_key1)
        state_list2 = KoreanChessV1.convert_state_list(state_key2)
        return np.sum(np.abs(np.array(state_list1) - np.array(state_list2)))

    def is_draw(self):
        for line in self.current_state:
            for piece in line:
                if piece is 0:
                    continue
                # todo: 상하고 마하고 구현해면 빼
                if piece[1] is not c.KING and piece[1] is not c.GUARDIAN:
                    return False
                    # todo: 포만 남았을경우 못넘는경우면 비긴걸로 계산
        return True

    def get_action(self, Q, state, i, is_red=False):
        action_list = self.get_action_list(state, is_red)
        action_cnt = len(action_list)
        if not Q or state not in Q:
            # if state is not in the Q, create state map and actions by state hash key
            Q[state] = np.zeros(action_cnt)

        if action_cnt < 1 or np.sum(Q[state]) == 0:
            q_state_key_list = {}
            for q_state_key in Q:
                diff_score = KoreanChessV1.compare_state(state, q_state_key)
                q_state_key_list[q_state_key] = diff_score

            sorted_q_state_list = sorted(q_state_key_list.items(), key=operator.itemgetter(1))
            for item in sorted_q_state_list:
                q_state = item[0]
                if np.sum(Q[q_state]) == 0:
                    continue
                q_max_action_no = np.argmax(Q[q_state])
                q_action_list = self.get_action_list(q_state, is_red)
                q_action = q_action_list[q_max_action_no]
                for i, action in enumerate(action_list):
                    if action['from_x'] == q_action['from_x'] \
                            and action['from_y'] == q_action['from_y'] \
                            and action['to_x'] == q_action['to_x'] \
                            and action['to_y'] == q_action['to_y']:
                        return i

        return np.argmax(Q[state] + np.random.randn(1, action_cnt) / (action_cnt * 10))

    def get_action_with_record(self, Q, state, record, is_red=False):
        if is_red:
            # reverse state
            action_list = self.state_list[KoreanChessV1.reverse_state_key(state)]['action_list']
        else:
            action_list = self.state_list[state]['action_list']

        if not Q or state not in Q:
            Q[state] = np.zeros(len(action_list))

        for i, action in enumerate(action_list):
            if action['from_x'] == record['from_x'] \
                    and action['from_y'] == record['from_y'] \
                    and action['to_x'] == record['to_x'] \
                    and action['to_y'] == record['to_y']:
                return i

        import json

        # return False
        # return np.argmax(Q[state] + np.random.randn(1, len(action_list)) / (i + 1))
        raise Exception("coudn't find record action\n" + json.dumps(action_list) + "\n" + json.dumps(record))

    def get_action_list(self, state, is_red=False):
        if is_red:
            # reverse state
            state_key = KoreanChessV1.reverse_state_key(state)
        else:
            state_key = state
        if state_key not in self.state_list:
            self.create_state(KoreanChessV1.convert_state_map(state_key), c.RED if is_red else c.BLUE)
        action_list = self.state_list[state_key]['action_list']
        return action_list

    @staticmethod
    def build_action_key(action):
        return str(action['from_x']) + ':' + str(action['from_y']) + ':' + str(action['to_x']) + ':' + str(action['to_y'])

    def get_action_test(self, Q, state, is_red=False):
        action_list = self.get_action_list(state, is_red)
        action_cnt = len(action_list)

        if not Q or state not in Q:
            Q[state] = np.zeros(action_cnt)
        if action_cnt < 1 or np.sum(Q[state]) == 0:
            q_state_key_list = {}
            for q_state_key in Q:
                diff_score = KoreanChessV1.compare_state(state, q_state_key)
                q_state_key_list[q_state_key] = diff_score

            sorted_q_state_list = sorted(q_state_key_list.items(), key=operator.itemgetter(1))
            for item in sorted_q_state_list:
                q_state = item[0]
                if np.sum(Q[q_state]) == 0:
                    continue
                q_max_action_no = np.argmax(Q[q_state])
                q_action_list = self.get_action_list(q_state, is_red)
                q_action = q_action_list[q_max_action_no]
                for i, action in enumerate(action_list):
                    if action['from_x'] == q_action['from_x'] \
                            and action['from_y'] == q_action['from_y'] \
                            and action['to_x'] == q_action['to_x'] \
                            and action['to_y'] == q_action['to_y']:
                        return i

        return np.argmax(Q[state] + np.random.randn(1, action_cnt) / (action_cnt * 10))
