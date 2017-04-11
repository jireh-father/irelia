# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import numbers
import operator
import os
import random
import sqlite3
import time

import numpy as np

from env.korean_chess import common
from env.env import Env
from env.korean_chess import piece_factory


class Core(Env):
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
        common.SOLDIER: 1,
        common.SANG: 2,
        common.GUARDIAN: 3,
        common.HORSE: 4,
        common.CANNON: 5,
        common.CAR: 6,
        common.KING: 46,
    }

    CONV_PIECE_LIST = {
        common.SOLDIER: '0.15',
        common.SANG: '0.29',
        common.GUARDIAN: '0.43',
        common.HORSE: '0.58',
        common.CANNON: '0.72',
        common.CAR: '0.86',
        common.KING: '1.0',
    }

    def __init__(self, properties):
        Env.__init__(self, properties)
        self.state_list = {}
        self.history = []

    def set_property(self, key, properties):
        self.properties[key] = properties

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

    def build_csv_data(self, state_key, color, records):
        blue_feature_map = []
        red_feature_map = []
        for piece in state_key.split(','):
            if piece.isdigit():
                blue_feature_map += ['0'] * int(piece)
                red_feature_map += ['0'] * int(piece)
            else:
                if piece[0] is 'b':
                    blue_feature_map.append(Core.CONV_PIECE_LIST[int(piece[1])])
                    red_feature_map.append('0')
                else:
                    red_feature_map.append(Core.CONV_PIECE_LIST[int(piece[1])])
                    blue_feature_map.append('0')
        if color is 'b':
            color_feature_map = ['0.5'] * 10 * 9
            y = records['y']
            x = records['x']
            to_y = records['to_y']
            to_x = records['to_x']
        else:
            color_feature_map = ['1'] * 10 * 9
            y = 9 - records['y']
            x = 8 - records['x']
            to_y = 9 - records['to_y']
            to_x = 8 - records['to_x']

        from_label_map = ['0'] * 10 * 9
        to_label_map = ['0'] * 10 * 9
        from_label_map[y * 9 + x] = '1'
        to_label_map[to_y * 9 + to_x] = '1'
        return blue_feature_map + red_feature_map + color_feature_map + from_label_map + to_label_map

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
        state_list = Core.convert_uncompressed_state_list(state_key)
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
                action_list += Core.get_piece_actions(state_map, x, y)

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
            reward = Core.REWARD_LIST[int(to_value[1])]

        # 이동 지점에 기존지점 말 저장
        state_map[to_y][to_x] = state_map[y][x]

        # 기존 지점 0으로 세팅
        state_map[y][x] = 0

        is_done = reward == Core.REWARD_LIST[common.KING] or Core.is_draw(state_map)

        # state_map 결과는 무조건 reverse해서 보내라
        return Core.reverse_state_map(state_map), reward, is_done, Core.is_draw(state_map)

    def is_losing_way(state_map, x, y, to_x, to_y, side):

        return False

    @staticmethod
    def reverse_state_map(state_map):
        reversed_map = np.array(list(reversed(np.array(state_map).flatten()))).reshape([-1, 9]).tolist()
        result_map = []
        for line in reversed_map:
            result_map.append([int(val) if val == u'0' else val for val in line])
        return result_map

    def reset(self):
        if self.properties['init_state']:
            default_map = self.properties['init_state']
            side = self.properties['init_side'] if self.properties['init_side'] else 'b'
        else:
            side = common.BLUE
            if not self.properties['position_type'] or self.properties['position_type'] == 'random':
                before_rand_position = random.randint(0, 3)
                after_rand_position = random.randint(0, 3)
                position_type_list = [Core.rand_position_list[before_rand_position],
                                      Core.rand_position_list[after_rand_position]]
            else:
                position_type_list = self.properties['position_type']

            default_map = copy.deepcopy(Core.default_state_map)

            for i, position_type in enumerate(position_type_list):
                if position_type not in Core.POSITION_TYPE_LIST:
                    raise Exception('position_type is invalid : ' + position_type)

                line_idx = -1 if i == 0 else 0

                default_map[line_idx] = Core.POSITION_TYPE_LIST[position_type][i]

        state_key = self.create_state(default_map, side)

        # self.print_map(state_key, side)

        # for action in self.state_list[state_key]['action_list']:
        #     print(action)

        return state_key

    def create_state(self, state_map, side):
        state_key = Core.convert_state_key(state_map)
        if state_key not in self.state_list:
            self.state_list[state_key] = {'state_map': state_map,
                                          'action_list': Core.get_actions(state_map, side), 'side': side}

        if side is common.RED:
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
        # if side is util.RED:
        #     state = self.reverse_state_key(state)
        #     map = Core.reverse_state_map(self.state_list[state]['state_map'])
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
        #     converted_line = [Core.PIECE_LIST[val] for val in line]
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
        if side is common.RED:
            state = Core.reverse_state_key(state)
            map = Core.reverse_state_map(self.state_list[state]['state_map'])
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
                converted_line = [Core.PIECE_LIST[val] for val in line]
                # sys.stdout.write('\r' + ' '.join(converted_line))
                print(' '.join(converted_line))
                # print('======================================================')

    def init_q_state(self, Q, state, is_red=False):
        if not Q or state not in Q:
            # if state is not in the Q, create state map and actions by state hash key
            if is_red:
                # reverse state
                action_cnt = len(self.state_list[Core.reverse_state_key(state)]['action_list'])
            else:
                action_cnt = len(self.state_list[state]['action_list'])
            Q[state] = np.zeros(action_cnt)

    @staticmethod
    def compare_state(state_key1, state_key2):
        state_list1 = Core.convert_state_list(state_key1)
        state_list2 = Core.convert_state_list(state_key2)
        return np.sum(np.abs(np.array(state_list1) - np.array(state_list2)))

    @staticmethod
    def is_draw(state_map):
        for line in state_map:
            for piece in line:
                if piece is 0:
                    continue
                # todo: 상하고 마하고 구현해면 빼
                if piece[1] is not common.KING and piece[1] is not common.GUARDIAN:
                    return False
                    # todo: 포만 남았을경우 못넘는경우면 비긴걸로 계산
        return True

    def step(self, action, state_key, is_red=False, use_es=False):
        opposite_side = common.BLUE if is_red else common.RED

        if action is False:
            if is_red is False:
                state_key = Core.reverse_state_key(state_key)
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
        if use_es:
            Core.insert_state_key(new_state_key, is_red)

        # return new_state, reward, is_done
        return new_state_key, reward, is_done, is_draw

    def insert_state_key(state_key, is_red=False):
        side = 'red' if is_red else 'blue'
        es = ES('52.79.135.2:80')
        result = es.search('i_irelia_state', 't_%s_state' % side, {
            "query": {
                "constant_score": {
                    "filter": {
                        "term": {
                            "state.keyword": state_key}
                    }
                }
            }
        })

        if result and 'hits' in result and result['hits']['total'] > 0:
            return True

        result = es.index('i_irelia_state', 't_%s_state' % side, {"state": state_key})
        return result and result['created'] == True

    def add_state_link(self, source_state, target_state, action):
        if source_state not in self.state_links:
            self.state_links[source_state] = {}

        if action not in self.state_links[source_state]:
            self.state_links[source_state][action] = target_state

    @staticmethod
    def reverse_state_key(state):
        return ','.join(list(reversed(state.split(','))))

    def reverse_state_key(self, state):
        return ','.join(list(reversed(state.split(','))))

    def get_action_es(self, state_key, side):
        if side == 'b':
            db_name = './q_blue.db'
            state_map = Core.convert_state_map(state_key)
        else:
            state_map = Core.reverse_state_map(Core.convert_state_map(state_key))
            db_name = './q_red.db'

        conn = sqlite3.connect(db_name)

        c = conn.cursor()

        c.execute("SELECT quality_json FROM t_quality WHERE state_key='" + state_key + "'")

        result = c.fetchone()
        actions = Core.get_actions(state_map, side)
        if result:
            if result[0] == '0':
                action_no = Core.similar_action_no(actions, state_key, side, c)
            else:
                q_values = json.loads(result[0])
                max_action = int(max(q_values.iteritems(), key=operator.itemgetter(1))[0])
                if len(actions) <= max_action:
                    action_no = Core.similar_action_no(actions, state_key, side, c)
                else:
                    action_no = max_action
        else:
            action_no = Core.similar_action_no(actions, state_key, side, c)

        return action_no

    def get_action(self, Q, state, i, is_red=False):
        action_list = self.get_action_list(state, is_red)
        action_cnt = len(action_list)
        if not Q or state not in Q:
            # if state is not in the Q, create state map and actions by state hash key
            Q[state] = np.zeros(action_cnt)

        if action_cnt < 1 or np.sum(Q[state]) == 0:
            q_state_key_list = {}
            for q_state_key in Q:
                diff_score = Core.compare_state(state, q_state_key)
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

    @staticmethod
    def get_action_list(state_key, color='b'):
        if color is 'r':
            # reverse state
            state_key = Core.reverse_state_key(state_key)
        else:
            state_key = state_key

        action_list = Core.get_actions(Core.convert_state_map(state_key), color)

    def get_action_list(self, state, is_red=False):
        if is_red:
            # reverse state
            state_key = Core.reverse_state_key(state)
        else:
            state_key = state
        if state_key not in self.state_list:
            self.create_state(Core.convert_state_map(state_key), common.RED if is_red else common.BLUE)
        action_list = self.state_list[state_key]['action_list']
        return action_list

    @staticmethod
    def build_action_key(action):
        return str(action['x']) + ':' + str(action['y']) + ':' + str(action['to_x']) + ':' + str(action['to_y'])

    def get_action_test(self, Q, state, is_red=False):
        action_list = self.get_action_list(state, is_red)
        action_cnt = len(action_list)

        if not Q or state not in Q:
            Q[state] = np.zeros(action_cnt)
        if action_cnt < 1 or np.sum(Q[state]) == 0:
            q_state_key_list = {}
            for q_state_key in Q:
                diff_score = Core.compare_state(state, q_state_key)
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

    @staticmethod
    def similar_action(actions, state_key, side, sqlite_cursor):
        # decomp key
        decomp_state_key = Core.decompress_state_key(state_key)
        # full text search for similar state key on elasticsearch
        es = ES('52.79.135.2:80')
        result = es.search('i_irelia_state', 't_blue_state' if side is 'b' else 't_red_state',
                           {
                               "query": {"match": {
                                   "state": decomp_state_key}}
                           })

        if not result or result['_shards']['failed'] > 0:
            return random.choice(actions)

        actions_map = {}
        for act in actions:
            actions_map[Core.build_action_key(act)] = True

        for item in result['hits']['hits']:
            similar_state = Core.compress_state_key(item['_source']['state'])
            sqlite_cursor.execute(
                "SELECT quality_json FROM t_quality WHERE state_key='" + Core.compress_state_key(
                    similar_state) + "'")

            q_json = sqlite_cursor.fetchone()
            if not q_json or q_json[0] == '0':
                continue

            similar_state_map = Core.convert_state_map(similar_state)
            if side == 'r':
                similar_state_map = Core.reverse_state_map(similar_state_map)
            similar_state_actions = Core.get_actions(similar_state_map, side)

            q_values = json.loads(q_json[0])

            q_values = sorted(q_values.items(), key=lambda x: (-x[1], x[0]))

            for q_value_tuple in q_values:
                # get action no
                action_no = int(q_value_tuple[0])
                q_value = q_value_tuple[1]
                if q_value <= 0:
                    break
                sim_action = similar_state_actions[action_no]
                if Core.build_action_key(sim_action) in actions_map:
                    return sim_action

        return random.choice(actions)

    @staticmethod
    def similar_action_no(actions, state_key, side, sqlite_cursor):
        # decomp key
        decomp_state_key = Core.decompress_state_key(state_key)
        # full text search for similar state key on elasticsearch
        es = ES('52.79.135.2:80')
        result = es.search('i_irelia_state', 't_blue_state' if side is 'b' else 't_red_state',
                           {
                               "query": {"match": {
                                   "state": decomp_state_key}}
                           })

        if not result or result['_shards']['failed'] > 0:
            return random.randint(0, len(actions) - 1)

        actions_map = {}
        for i, act in enumerate(actions):
            actions_map[Core.build_action_key(act)] = i

        for item in result['hits']['hits']:
            similar_state = Core.compress_state_key(item['_source']['state'])
            sqlite_cursor.execute(
                "SELECT quality_json FROM t_quality WHERE state_key='" + Core.compress_state_key(
                    similar_state) + "'")

            q_json = sqlite_cursor.fetchone()
            if not q_json or q_json[0] == '0':
                continue

            similar_state_map = Core.convert_state_map(similar_state)
            if side == 'r':
                similar_state_map = Core.reverse_state_map(similar_state_map)
            similar_state_actions = Core.get_actions(similar_state_map, side)

            q_values = json.loads(q_json[0])

            q_values = sorted(q_values.items(), key=lambda x: (-x[1], x[0]))

            for q_value_tuple in q_values:
                # get action no
                action_no = int(q_value_tuple[0])
                q_value = q_value_tuple[1]
                if q_value <= 0:
                    break
                sim_action = similar_state_actions[action_no]
                sim_action_key = Core.build_action_key(sim_action)
                if sim_action_key in actions_map:
                    return actions_map[sim_action_key]

        return random.randint(0, len(actions) - 1)

    @staticmethod
    def get_q_from_es(state_key, side):
        if side == 'b':
            db_name = './q_blue.db'
        else:
            db_name = './q_red.db'

        conn = sqlite3.connect(db_name)

        c = conn.cursor()

        c.execute("SELECT quality_json FROM t_quality WHERE state_key='" + state_key + "'")

        result = c.fetchone()

        if result:
            if result[0] == '0':
                return None
            else:
                return json.loads(result[0])
        else:
            return None
