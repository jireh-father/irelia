# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import time

from game import korean_chess_constant as c
from game import korean_chess_util as u
from colorama import Fore
import os

if os.name == "posix":
    empty_str = "--"
    empty_str_kor = "-----"
elif os.name == "nt":
    empty_str = "--"
    empty_str_kor = "------"
else:
    empty_str = "--"
    empty_str_kor = "----"


class KoreanChessV1:
    BLUE_COLOR = Fore.BLUE
    RED_COLOR = Fore.RED
    LAST_COLOR = Fore.MAGENTA
    EMPTY_COLOR = Fore.WHITE
    PIECE_MAP_COLOR = \
        {c.R_SD: RED_COLOR + '졸' + EMPTY_COLOR, c.R_SG: RED_COLOR + '상' + EMPTY_COLOR,
         c.R_GD: RED_COLOR + '사' + EMPTY_COLOR, c.R_HS: RED_COLOR + '마' + EMPTY_COLOR,
         c.R_CN: RED_COLOR + '포' + EMPTY_COLOR, c.R_CR: RED_COLOR + '차' + EMPTY_COLOR,
         c.R_KG: RED_COLOR + '궁' + EMPTY_COLOR,
         c.B_SD: Fore.BLUE + '졸' + EMPTY_COLOR, c.B_SG: Fore.BLUE + '상' + EMPTY_COLOR,
         c.B_GD: Fore.BLUE + '사' + EMPTY_COLOR, c.B_HS: Fore.BLUE + '마' + EMPTY_COLOR,
         c.B_CN: Fore.BLUE + '포' + EMPTY_COLOR, c.B_CR: Fore.BLUE + '차' + EMPTY_COLOR,
         c.B_KG: Fore.BLUE + '궁' + EMPTY_COLOR,
         0: EMPTY_COLOR + empty_str + EMPTY_COLOR}

    PIECE_MAP_COLOR_MOVED = \
        {c.R_SD: LAST_COLOR + '졸' + EMPTY_COLOR, c.R_SG: LAST_COLOR + '상' + EMPTY_COLOR,
         c.R_GD: LAST_COLOR + '사' + EMPTY_COLOR, c.R_HS: LAST_COLOR + '마' + EMPTY_COLOR,
         c.R_CN: LAST_COLOR + '포' + EMPTY_COLOR, c.R_CR: LAST_COLOR + '차' + EMPTY_COLOR,
         c.R_KG: LAST_COLOR + '궁' + EMPTY_COLOR,
         c.B_SD: LAST_COLOR + '졸' + EMPTY_COLOR, c.B_SG: LAST_COLOR + '상' + EMPTY_COLOR,
         c.B_GD: LAST_COLOR + '사' + EMPTY_COLOR, c.B_HS: LAST_COLOR + '마' + EMPTY_COLOR,
         c.B_CN: LAST_COLOR + '포' + EMPTY_COLOR, c.B_CR: LAST_COLOR + '차' + EMPTY_COLOR,
         c.B_KG: LAST_COLOR + '궁' + EMPTY_COLOR,
         0: LAST_COLOR + empty_str + EMPTY_COLOR}

    PIECE_MAP_KOR = \
        {c.R_SD: '졸(홍)', c.R_SG: '상(홍)', c.R_GD: '사(홍)', c.R_HS: '마(홍)', c.R_CN: '포(홍)', c.R_CR: '차(홍)', c.R_KG: '궁(홍)',
         c.B_SD: '졸(청)', c.B_SG: '상(청)', c.B_GD: '사(청)', c.B_HS: '마(청)', c.B_CN: '포(청)', c.B_CR: '차(청)', c.B_KG: '궁(청)',
         0: empty_str_kor}

    PIECE_MAP_KOR_MOVED = \
        {c.R_SD: '졸(V)', c.R_SG: '상(V)', c.R_GD: '사(V)', c.R_HS: '마(V)', c.R_CN: '포(V)', c.R_CR: '차(V)', c.R_KG: '궁(V)',
         c.B_SD: '졸(V)', c.B_SG: '상(V)', c.B_GD: '사(V)', c.B_HS: '마(V)', c.B_CN: '포(V)', c.B_CR: '차(V)', c.B_KG: '궁(V)',
         0: empty_str_kor}

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
        self.max_reward = 1
        self.data_format = None
        self.check_repeat = True
        self.print_mcts_history = None
        self.use_color_print = None
        self.action_history = {c.BLUE: [], c.RED: []}

    def reset(self):
        self.interval = 0
        self.use_check = True
        self.check_repeat = 5
        self.limit_step = 200
        self.max_reward = 1
        self.print_mcts_history = False
        self.use_color_print = False
        if self.properties:
            if "interval" in self.properties:
                self.interval = self.properties["interval"]
            if "use_check" in self.properties:
                self.use_check = self.properties["use_check"]
            if "limit_step" in self.properties:
                self.limit_step = self.properties["limit_step"]
            if "max_reward" in self.properties:
                self.max_reward = self.properties["max_reward"]
            if "data_format" in self.properties:
                self.data_format = self.properties["data_format"]
            if "check_repeat" in self.properties:
                self.check_repeat = self.properties["check_repeat"]
            if "print_mcts_history" in self.properties:
                self.print_mcts_history = self.properties["print_mcts_history"]
            if "use_color_print" in self.properties:
                self.use_color_print = self.properties["use_color_print"]

        if self.properties and "init_state" in self.properties:
            self.current_state, self.current_turn = u.encode_state(self.properties["init_state"], self.data_format)
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

        return u.decode_state(self.current_state, self.current_turn, self.data_format)

    def step(self, action):
        # validate action
        if not u.validate_action(action, self.current_state, self.current_turn, self.next_turn, self.use_check,
                                 self.action_history, self.check_repeat):
            raise Exception("Invalid action :%s" % action)
        self.action_history[self.current_turn].append(action)
        to_x = action['to_x']
        to_y = action['to_y']
        from_x = action['from_x']
        from_y = action['from_y']

        # check? 장군
        if self.use_check:
            is_check = u.is_check(self.current_state, from_x, from_y, to_x, to_y, self.current_turn)
        else:
            is_check = False

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

        # checkmate?
        is_checkmate = False
        if is_check:
            is_checkmate = u.is_checkmate(self.current_state, self.current_turn)
            if is_checkmate:
                reward = c.REWARD_LIST[c.KING]

        # draw?
        is_draw = u.is_draw(self.current_state)

        # todo: 장군, 외통수 상태편 수둘때도 고려해서 수정하기
        # todo: 먹힌말 print
        # todo: 빅장 - count, win or lose by count(점수에 의한 승부 정리)
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
        reward = (float(reward) / c.REWARD_LIST[c.KING] * self.max_reward)

        info = {"is_check": is_check,
                "over_limit_step": self.current_step >= self.limit_step,
                "is_draw": is_draw}

        # who's winner?
        if info["over_limit_step"] or info["is_draw"]:
            if self.blue_score > self.red_score:
                winner = c.BLUE
            elif self.blue_score < self.red_score:
                winner = c.RED
            else:
                winner = None
        else:
            winner = 'b' if self.current_turn == 'r' else 'b'
        info["winner"] = winner

        return u.decode_state(self.current_state, self.current_turn, self.data_format), reward, is_game_over, info

    def print_env(self, is_check=False, is_checkmate=False, to_x=10, to_y=10, done=False, is_draw=False, state=None):
        if state is None:
            by_mcts = False
            state = self.current_state
            turn = self.current_turn
        else:
            if not self.print_mcts_history:
                return
            by_mcts = True
            state, turn = u.encode_state(state, self.data_format)
        if self.interval > 0:
            time.sleep(self.interval)
        if turn == c.BLUE:
            print("%s %s : %d" % ("BLUE", "Turn", self.current_step))
        else:
            print("%s %s : %d" % ("RED", "Turn", self.current_step))
        if not by_mcts:
            print("Score [ BLUE : %f ] [ RED : %f ]" % (self.blue_score, self.red_score))
        if self.use_color_print:
            piece_map = KoreanChessV1.PIECE_MAP_COLOR
            piece_map_moved = KoreanChessV1.PIECE_MAP_COLOR_MOVED
        else:
            piece_map = KoreanChessV1.PIECE_MAP_KOR
            piece_map_moved = KoreanChessV1.PIECE_MAP_KOR_MOVED
        print("  " + piece_map[0].join(["%d" % col_idx for col_idx in range(0, 9)]) + "  X")
        for i, line in enumerate(state):
            if to_y == i:
                line = [piece_map_moved[piece] if j == to_x else
                        piece_map[piece] for j, piece in enumerate(line)]
            else:
                line = [piece_map[piece] for piece in line]
            print("%d %s" % (i, ' '.join(line)))
        print("Y")
        if not by_mcts:
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

    def get_all_actions(self, state=None):
        turn = None
        if state is not None:
            state, turn = u.encode_state(state, self.data_format)
        return u.get_all_actions(self.current_state if not state else state, self.current_turn if not turn else turn)

    def encode_action(self, action):
        action_from = action["from_y"] * 9 + action["from_x"]

        action_to = action["to_y"] * 9 + action["to_x"]
        return [action_from, action_to]

    def is_over(self, state):
        state, turn = u.encode_state(state, self.data_format)
        num_kings = 0
        for line in state:
            for piece in line:
                if piece != 0 and int(piece[1]) == c.KING:
                    num_kings += 1
                    if num_kings == 2:
                        return False
        return True

    def simulate(self, state, action, return_info=True):
        state, turn = u.encode_state(state, self.data_format)
        turn = c.RED if turn == c.BLUE else c.BLUE
        to_x = action['to_x']
        to_y = action['to_y']
        from_x = action['from_x']
        from_y = action['from_y']

        reward = 0
        if state[to_y][to_x] != 0 and return_info:
            reward = c.REWARD_LIST[int(state[to_y][to_x][1])] / c.KING

        state[to_y][to_x] = state[from_y][from_x]
        state[from_y][from_x] = 0

        # decode and return state
        is_game_over = reward >= c.REWARD_LIST[c.KING]

        info = {"is_game_over": is_game_over, "winner": None, "reward": reward}

        # who's winner?
        if is_game_over:
            info["winner"] = 'b' if self.current_turn == 'r' else 'b'

        if return_info:
            return u.decode_state(state, turn, self.data_format), info
        else:
            return u.decode_state(state, turn, self.data_format)

    def convert_action_probs_to_policy_probs(self, actions, action_probs):
        policy_probs = [.0] * 90
        for i, prob in enumerate(action_probs):
            action = actions[i]
            action = self.encode_action(action)
            half_prob = prob / 2
            policy_probs[action[0]] += half_prob
            policy_probs[action[1]] += half_prob
        return policy_probs
