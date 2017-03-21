# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from env.korean_chess import common
from env.korean_chess import cannon, common
from env.korean_chess import car
from env.korean_chess import guardian
from env.korean_chess import horse
from env.korean_chess import king
from env.korean_chess import sang
from env.korean_chess import soldier


def get_actions(state_map, x, y):
    piece_num = int(state_map[y][x][1])
    if piece_num == common.KING:
        return king.get_actions(state_map, x, y)
    elif piece_num == common.SOLDIER:
        return soldier.get_actions(state_map, x, y)
    elif piece_num == common.SANG:
        return sang.get_actions(state_map, x, y)
    elif piece_num == common.GUARDIAN:
        return guardian.get_actions(state_map, x, y)
    elif piece_num == common.HORSE:
        return horse.get_actions(state_map, x, y)
    elif piece_num == common.CANNON:
        return cannon.get_actions(state_map, x, y)
    elif piece_num == common.CAR:
        return car.get_actions(state_map, x, y)


def reverse_state_map(state_map):
    return state_map


def action(state_map, action):
    copy_state_map = copy.deepcopy(state_map)
    if action['piece_type'] == common.KING:
        new_state_map, reward, is_done = king.action(copy_state_map, action)
    elif action['piece_type'] == common.SOLDIER:
        new_state_map, reward, is_done = soldier.action(copy_state_map, action)
    elif action['piece_type'] == common.SANG:
        new_state_map, reward, is_done = sang.action(copy_state_map, action)
    elif action['piece_type'] == common.GUARDIAN:
        new_state_map, reward, is_done = guardian.action(copy_state_map, action)
    elif action['piece_type'] == common.HORSE:
        new_state_map, reward, is_done = horse.action(copy_state_map, action)
    elif action['piece_type'] == common.CANNON:
        new_state_map, reward, is_done = cannon.action(copy_state_map, action)
    elif action['piece_type'] == common.CAR:
        new_state_map, reward, is_done = car.action(copy_state_map, action)

    # state_map 결과는 무조건 reverse해서 보내라
    return reverse_state_map(new_state_map), reward, is_done
