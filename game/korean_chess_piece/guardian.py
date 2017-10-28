# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env import korean_chess_util as kcu


def get_actions(state_map, x, y):
    piece = state_map[y][x]

    side = piece[0]

    action_list = []

    piece_type = kcu.GUARDIAN

    # 대각선 오른쪽 전진 길 체크
    if ((y == 9 and x == 3) or (y == 8 and x == 4)) and \
            not kcu.is_our_side(state_map, x + 1, y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x + 1, y - 1, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x + 1, 'to_y': y - 1})

    # 대각선 왼쪽 전진 길 체크
    if ((y == 9 and x == 5) or (y == 8 and x == 4)) and \
            not kcu.is_our_side(state_map, x - 1, y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x - 1, y - 1, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x - 1, 'to_y': y - 1})

    # 대각선 오른쪽 후진 길 체크
    if ((y == 7 and x == 3) or (y == 8 and x == 4)) and \
            not kcu.is_our_side(state_map, x + 1, y + 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x + 1, y + 1, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x + 1, 'to_y': y + 1})

    # 대각선 왼쪽 후진 길 체크
    if ((y == 7 and x == 5) or (y == 8 and x == 4)) and \
            not kcu.is_our_side(state_map, x - 1, y + 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x - 1, y + 1, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x - 1, 'to_y': y + 1})

    # 전진 길 체크
    if y in (8, 9) and not kcu.is_our_side(state_map, x, y - 1, side):
        # and   not kcu.is_losing_way(state_map, x, y, x, y - 1, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x, 'to_y': y - 1})

    # 오른쪽 길 체크
    if x in (3, 4) and not kcu.is_our_side(state_map, x + 1, y, side):
        # and  not kcu.is_losing_way(state_map, x, y, x + 1, y, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x + 1, 'to_y': y})

    # 왼쪽 길 체크
    if x in (4, 5) and not kcu.is_our_side(state_map, x - 1, y, side):
        # and   not kcu.is_losing_way(state_map, x, y, x - 1, y, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x - 1, 'to_y': y})

    # 후진 길 체크
    if y in (7, 8) and not kcu.is_our_side(state_map, x, y + 1, side):
        # and   not kcu.is_losing_way(state_map, x, y, x, y + 1, side):
        action_list.append(
            {'from_x': x, 'from_y': y, 'to_x': x, 'to_y': y + 1})

    return action_list
