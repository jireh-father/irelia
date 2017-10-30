# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from game import korean_chess_constant as kcu


def get_actions(state_map, x, y):
    piece = state_map[y][x]
    side = piece[0]

    action_list = []

    piece_type = kcu.SOLDIER

    # 대각선 오른쪽 길 체크
    if ((y == 1 and x == 4) or (y == 2 and x == 3)) and \
            not kcu.is_our_side(state_map, x + 1, y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x + 1, y - 1, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x + 1, 'to_y': y - 1})

    # 대각선 왼쪽 길 체크
    if ((y == 1 and x == 4) or (y == 2 and x == 5)) and \
            not kcu.is_our_side(state_map, x - 1, y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x - 1, y - 1, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x - 1, 'to_y': y - 1})

    # 전진 길 체크
    if y != 0 and not kcu.is_our_side(state_map, x, y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x, y - 1, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x, 'to_y': y - 1})

    # 오른쪽 길 체크
    if x != 8 and not kcu.is_our_side(state_map, x + 1, y, side):
        # and   not kcu.is_losing_way(state_map, x, y, x + 1, y, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x + 1, 'to_y': y})

    # 왼쪽 길 체크
    if x != 0 and not kcu.is_our_side(state_map, x - 1, y, side):
        # and   not kcu.is_losing_way(state_map, x, y, x - 1, y, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x - 1, 'to_y': y})

    return action_list
