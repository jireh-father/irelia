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
        self.next_turn = None
        self.is_checkmate = None

    def reset(self):
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
        self.is_checkmate = False

        # setting state
        current_state = copy.deepcopy(KoreanChessV1.default_state)
        for i, position_type in enumerate(position_type_list):
            if not KoreanChessV1.POSITION_TYPE_LIST[position_type]:
                raise Exception('position_type is invalid : ' + str(position_type))

            line_idx = -1 if i == 0 else 0

            current_state[line_idx] = KoreanChessV1.POSITION_TYPE_LIST[position_type][i]
        self.current_state = current_state

        # print environment
        self.print_env()

        return u.decode_state(self.current_state)

    def step(self, action):
        # validate action
        if not u.validate_action(action, self.current_state, self.current_turn, self.next_turn):
            raise Exception("Invalid action :%s" % action)
        to_x = action['to_x']
        to_y = action['to_y']
        from_x = action['from_x']
        from_y = action['from_y']

        # checkmate?
        self.is_checkmate = u.is_checkmate(self.current_state, from_x, from_y, to_x, to_y, self.current_turn)

        # reward
        to_piece = self.current_state[to_y][to_x]
        reward = 0 if to_piece == 0 else KoreanChessV1.REWARD_LIST[int(to_piece[1])]

        # move
        self.current_state[to_y][to_x] = self.current_state[from_y][from_x]
        self.current_state[from_y][from_x] = 0

        # draw?
        is_draw = u.is_draw(self.current_state)
        # todo: we tong su
        # todo: repeat limit
        # todo: count, win or lose by count
        # todo: turn count limit

        # todo: 장군했는데 상대가 왕이 먹히는 수를 두는지 체크하는거 추가
        # done?
        done = (reward == KoreanChessV1.REWARD_LIST[c.KING] or is_draw)

        # change turn
        old_turn = self.current_turn
        self.current_turn = self.next_turn
        self.next_turn = old_turn

        # print env
        self.print_env()

        # decode and return state
        return u.decode_state(self.current_state), reward, done, self.is_checkmate

    def print_env(self, interval=0):
        if interval > 0:
            time.sleep(0.5)
        if self.current_turn == c.BLUE:
            print("%s %s" % ("BLUE", "Turn"))
        else:
            print("%s %s" % ("RED", "Turn"))
        print("Y  X " + KoreanChessV1.PIECE_MAP_KOR[0].join(["%d" % col_idx for col_idx in range(0, 9)]))
        for i, line in enumerate(self.current_state):
            line = [KoreanChessV1.PIECE_MAP_KOR[piece] for piece in line]
            print("%d %s" % (i, ' '.join(line)))

        if self.is_checkmate:
            print("Checkmate!!")
        print('======================================================')
