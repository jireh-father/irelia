from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env.korean_chess_piece import cannon
from env.korean_chess_piece import car
from env.korean_chess_piece import guardian
from env.korean_chess_piece import horse
from env.korean_chess_piece import king
from env.korean_chess_piece import sang
from env.korean_chess_piece import soldier
from env import korean_chess_util

import copy


def get_actions(state_map, x, y):
    piece_num = int(state_map[y][x][1])
    if piece_num == korean_chess_util.KING:
        return king.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.SOLDIER:
        return soldier.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.SANG:
        return sang.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.GUARDIAN:
        return guardian.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.HORSE:
        return horse.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.CANNON:
        return cannon.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.CAR:
        return car.get_actions(state_map, x, y)


def reverse_state_map(state_map):
    return state_map


def action(state_map, action):
    copy_state_map = copy.deepcopy(state_map)
    if action['piece_type'] == korean_chess_util.KING:
        new_state_map, reward, is_done = king.action(copy_state_map, action)
    elif action['piece_type'] == korean_chess_util.SOLDIER:
        new_state_map, reward, is_done = soldier.action(copy_state_map, action)
    elif action['piece_type'] == korean_chess_util.SANG:
        new_state_map, reward, is_done = sang.action(copy_state_map, action)
    elif action['piece_type'] == korean_chess_util.GUARDIAN:
        new_state_map, reward, is_done = guardian.action(copy_state_map, action)
    elif action['piece_type'] == korean_chess_util.HORSE:
        new_state_map, reward, is_done = horse.action(copy_state_map, action)
    elif action['piece_type'] == korean_chess_util.CANNON:
        new_state_map, reward, is_done = cannon.action(copy_state_map, action)
    elif action['piece_type'] == korean_chess_util.CAR:
        new_state_map, reward, is_done = car.action(copy_state_map, action)

    # state_map 결과는 무조건 reverse해서 보내라
    return reverse_state_map(new_state_map), reward, is_done
