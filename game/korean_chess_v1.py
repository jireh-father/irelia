# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import time

from game import korean_chess_constant as c
from game import korean_chess_util as u


class KoreanChessV1:
    PIECE_MAP_KOR = \
        {c.R_SD: '졸(홍)', c.R_SG: '상(홍)', c.R_GD: '사(홍)', c.R_HS: '마(홍)', c.R_CN: '포(홍)', c.R_CR: '차(홍)', c.R_KG: '궁(홍)',
         c.B_SD: '졸(청)', c.B_SG: '상(청)', c.B_GD: '사(청)', c.B_HS: '마(청)', c.B_CN: '포(청)', c.B_CR: '차(청)', c.B_KG: '궁(청)',
         0: '------'}

    PIECE_MAP_KOR_NO_COLOR = \
        {c.R_SD: '졸(V)', c.R_SG: '상(V)', c.R_GD: '사(V)', c.R_HS: '마(V)', c.R_CN: '포(V)', c.R_CR: '차(V)', c.R_KG: '궁(V)',
         c.B_SD: '졸(V)', c.B_SG: '상(V)', c.B_GD: '사(V)', c.B_HS: '마(V)', c.B_CN: '포(V)', c.B_CR: '차(V)', c.B_KG: '궁(V)',
         0: '------'}

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

    def __init__(self, properties):
        self.properties = properties
        self.current_state = None
        self.current_turn = None
        self.next_turn = None
        self.red_score = None
        self.blue_score = None
        self.interval = None
        self.current_step = None
        self.use_check = None
        self.red_catch_list = []
        self.blue_catch_list = []
        self.limit_step = 200

    def reset(self):
        self.interval = 0
        self.use_check = True
        self.limit_step = 200
        if self.properties:
            if "interval" in self.properties:
                self.interval = self.properties["interval"]
            if "use_check" in self.properties:
                self.use_check = self.properties["use_check"]
            if "limit_step" in self.properties:
                self.limit_step = self.properties["limit_step"]

        if self.properties and "init_state" in self.properties:
            self.current_state, self.current_turn = u.encode_state(self.properties["init_state"])
            self.next_turn = c.RED if self.current_turn == c.BLUE else c.BLUE
        else:
            if not self.properties or (
                      "position_type" not in self.properties or self.properties['position_type'] == 'random'):
                # random position
                blue_rand_position = random.randint(0, 3)
                red_rand_position = random.randint(0, 3)
                position_type_list = [blue_rand_position, red_rand_position]
            else:
                position_type_list = self.properties['position_type']

            # setting turn
            self.current_turn = c.BLUE
            self.next_turn = c.RED

            # setting state
            current_state = copy.deepcopy(KoreanChessV1.default_state)
            for i, position_type in enumerate(position_type_list):
                if not KoreanChessV1.POSITION_TYPE_LIST[position_type]:
                    raise Exception('position_type is invalid : ' + str(position_type))

                line_idx = -1 if i == 0 else 0

                current_state[line_idx] = copy.deepcopy(KoreanChessV1.POSITION_TYPE_LIST[position_type][i])
            self.current_state = current_state

        # set scores
        self.blue_score = c.get_score(self.current_state, self.current_turn)
        self.red_score = c.get_score(self.current_state, self.next_turn)
        self.current_step = 0

        # print environment
        self.print_env()

        return u.decode_state(self.current_state, self.current_turn)

    def step(self, action):
        # validate action
        if not u.validate_action(action, self.current_state, self.current_turn, self.next_turn, self.use_check):
            raise Exception("Invalid action :%s" % action)
        to_x = action['to_x']
        to_y = action['to_y']
        from_x = action['from_x']
        from_y = action['from_y']

        # check? 장군
        is_check = u.is_check(self.current_state, from_x, from_y, to_x, to_y, self.current_turn)

        # reward
        to_piece = self.current_state[to_y][to_x]
        reward = 0 if to_piece == 0 else c.REWARD_LIST[int(to_piece[1])]
        if reward > 0 and reward < c.REWARD_LIST[c.KING]:
            if self.current_turn == c.BLUE:
                self.blue_catch_list.append(to_piece)
                self.red_score -= reward
            else:
                self.red_catch_list.append(to_piece)
                self.blue_score -= reward

        # move
        self.current_state[to_y][to_x] = self.current_state[from_y][from_x]
        self.current_state[from_y][from_x] = 0
        self.current_step += 1

        # checkmate? 외통수
        is_checkmate = False
        if is_check:
            is_checkmate = u.is_checkmate(self.current_state, self.current_turn)
            if is_checkmate:
                reward = c.REWARD_LIST[c.KING]

        # draw?
        is_draw = u.is_draw(self.current_state)
        # reverse 랑 액션쪽 하는거 다 바꾸기 그리고 내편 상대편으로 각 기능 다 시뮬레이션 해보기

        # todo: 먹힌말 print
        # todo: repeat limit(반복수)
        # todo: count, win or lose by count(점수에 의한 승부 정리)
        # todo: 빅장
        # todo: 장군, 가능한 actions, 외통수 등 기능 테스트

        # done?
        done = reward >= c.REWARD_LIST[c.KING]

        # change turn
        old_turn = self.current_turn
        self.current_turn = self.next_turn
        self.next_turn = old_turn

        # print env
        self.print_env(is_check, is_checkmate, to_x, to_y, done, is_draw)

        # decode and return state
        is_game_over = (done or is_draw or self.current_step >= self.limit_step)
        reward = (float(reward) / c.REWARD_LIST[c.KING] * 1)
        info = {"is_check": is_check,
                "over_limit_step": self.current_step >= self.limit_step,
                "is_draw": is_draw}
        return u.decode_state(self.current_state, self.current_turn), reward, is_game_over, info

    def print_env(self, is_check=False, is_checkmate=False, to_x=10, to_y=10, done=False, is_draw=False):
        if self.interval > 0:
            time.sleep(self.interval)
        if self.current_turn == c.BLUE:
            print("%s %s : %d" % ("BLUE", "Turn", self.current_step))
        else:
            print("%s %s : %d" % ("RED", "Turn", self.current_step))
        print("Score [ BLUE : %f ] [ RED : %f ]" % (self.blue_score, self.red_score))
        print("Y  X " + KoreanChessV1.PIECE_MAP_KOR[0].join(["%d" % col_idx for col_idx in range(0, 9)]))
        for i, line in enumerate(self.current_state):
            if to_y == i:
                line = [KoreanChessV1.PIECE_MAP_KOR_NO_COLOR[piece] if j == to_x else
                        KoreanChessV1.PIECE_MAP_KOR[piece] for j, piece in enumerate(line)]
            else:
                line = [KoreanChessV1.PIECE_MAP_KOR[piece] for piece in line]
            print("%d %s" % (i, ' '.join(line)))

        if is_check:
            print("Check!!")
            if is_checkmate:
                print("Checkmate!!")
        if done:
            if self.next_turn == c.BLUE:
                print("BLUE WIN")
            else:
                print("RED WIN")
        if is_draw:
            print("draw!!")

        if self.current_step >= self.limit_step:
            print("")
        print('======================================================')

    def get_all_actions(self):

        return u.get_all_actions(self.current_state, self.current_turn)

    def reverse_action(self, action):
        return u.reverse_actions([action])[0]

    def encode_action(action):
        action_from = action["from_y"] * 9 + action["from_x"]

        action_to = action["to_y"] * 9 + action["to_x"]
        return [action_from, action_to]
