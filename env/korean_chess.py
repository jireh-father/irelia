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
import numbers

from env.env import Env
from env.korean_chess_piece import piece_factory
from env import korean_chess_util as kcu
import operator


class KoreanChess(Env):
    PIECE_LIST = {'r1': '졸(홍)', 'r2': '상(홍)', 'r3': '사(홍)', 'r4': '마(홍)', 'r5': '포(홍)', 'r6': '차(홍)', 'r7': '궁(홍)',
                  'b1': '졸(청)', 'b2': '상(청)', 'b3': '사(청)', 'b4': '마(청)', 'b5': '포(청)', 'b6': '차(청)', 'b7': '궁(청)',
                  0: '------'}

    state_list = {}
    state_links = {}
    history = []
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
        self.history = []

    def set_property(self, key, properties):
        self.properties[key] = properties

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
        self.history.append({'x': x, 'y': y, 'to_x': to_x, 'to_y': to_y})

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

    def is_losing_way(state_map, x, y, to_x, to_y, side):

        return False

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
                  is_draw=False, blue_win_cnt=0, red_win_cnt=0, Q1=None, Q2=None, file=None, line=None):
        if turn % 60 is not 0:
            return
        # time.sleep(0.5)
        # if os.name == 'nt':
        #     os.system('cls')
        # else:
        #     os.system('clear')
        # sys.stdout.flush()
        # if side is kcu.RED:
        #     state = self.reverse_state_key(state)
        #     map = self.reverse_state_map(self.state_list[state]['state_map'])
        # else:
        #     map = self.state_list[state]['state_map']
        print(
            'EPISODE {:s}, TURN {:d}, BLUE REWARD {:d}, RED REWARD {:d}'.format(str(episode), turn, blue_reward_episode,
                                                                                red_reward_episode))
        # if Q1 and Q2:
        #     print('Q1 COUNT {:d}, Q2 COUNT {:d}'.format(len(Q1), len(Q2)))
        #     print('TOTAL BLUE WIN {:d}, TOTAL RED WIN {:d}, TOTAL STATE COUNT {:d}'.format(blue_win_cnt, red_win_cnt,
        #                                                                                    len(self.state_list)))
        #
        # if is_draw:
        #     print('draw')
        # elif done_side:
        #     print('WiNNER {:s}'.format(done_side))
        # else:
        #     print('running : ' + side)
        # if file and line:
        #     print(file, line)
        #
        # for line in map:
        #     converted_line = [KoreanChess.PIECE_LIST[val] for val in line]
        #     # sys.stdout.write('\r' + ' '.join(converted_line))
        #     print(' '.join(converted_line))
        #     # print('======================================================')

    def print_map_for_test(self, state, side, episode=0, turn=0, blue_reward_episode=0, red_reward_episode=0,
                           done_side=False,
                           is_draw=False, blue_win_cnt=0, red_win_cnt=0, Q1=None, Q2=None):
        # if turn % 20 is not 0:
        #     return
        time.sleep(0.2)
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
            'EPISODE {:s}, TURN {:d}, BLUE REWARD {:d}, RED REWARD {:d}'.format(str(episode), turn, blue_reward_episode,
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
                # print('======================================================')

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
        state_map = []
        for piece in state_key.split(','):
            if piece.isdigit():
                state_map += [0] * int(piece)
            else:
                state_map.append(piece)
        return np.array(state_map).reshape([-1, 9]).tolist()

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
        state_list = KoreanChess.convert_uncompressed_state_list(state_key)
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
    def compare_state(state_key1, state_key2):
        state_list1 = KoreanChess.convert_state_list(state_key1)
        state_list2 = KoreanChess.convert_state_list(state_key2)
        return np.sum(np.abs(np.array(state_list1) - np.array(state_list2)))

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

        # add
        self.add_state_link(state_key, new_state_key, action)

        # print next state
        # self.print_map(new_state_key, opposite_side)

        # return new_state, reward, is_done
        return new_state_key, reward, is_done, is_draw

    def add_state_link(self, source_state, target_state, action):
        if source_state not in self.state_links:
            self.state_links[source_state] = {}

        if action not in self.state_links[source_state]:
            self.state_links[source_state][action] = target_state

    def reverse_state_key(self, state):
        return ','.join(list(reversed(state.split(','))))

    def get_action(self, Q, state, i, is_red=False):
        if not Q or state not in Q:
            # if state is not in the Q, create state map and actions by state hash key
            action_cnt = len(self.get_action_list(state, is_red))
            Q[state] = np.zeros(action_cnt)

        action_cnt = len(Q[state])
        if action_cnt < 1:
            return False
        else:
            return np.argmax(Q[state] + np.random.randn(1, action_cnt) / (i + 1))

    def get_action_with_record(self, Q, state, record, is_red=False):
        if is_red:
            # reverse state
            action_list = self.state_list[self.reverse_state_key(state)]['action_list']
        else:
            action_list = self.state_list[state]['action_list']

        if not Q or state not in Q:
            Q[state] = np.zeros(len(action_list))

        for i, action in enumerate(action_list):
            if action['x'] == record['x'] \
              and action['y'] == record['y'] \
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
            state_key = self.reverse_state_key(state)
        else:
            state_key = state
        if state_key not in self.state_list:
            self.create_state(KoreanChess.convert_state_map(state_key), kcu.RED if is_red else kcu.BLUE)
        action_list = self.state_list[state_key]['action_list']
        return action_list

    def get_action_test(self, Q, state, is_red=False):
        action_list = self.get_action_list(state, is_red)
        action_cnt = len(action_list)

        if not Q or state not in Q:
            Q[state] = np.zeros(action_cnt)

        if action_cnt < 1 or np.sum(Q[state]) == 0:
            q_state_key_list = {}
            for q_state_key in Q:
                diff_score = KoreanChess.compare_state(state, q_state_key)
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
                    if action['x'] == q_action['x'] \
                      and action['y'] == q_action['y'] \
                      and action['to_x'] == q_action['to_x'] \
                      and action['to_y'] == q_action['to_y']:
                        return i

        return np.argmax(Q[state] + np.random.randn(1, action_cnt) / (action_cnt * 10))
