# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env import korean_chess_util as kcu


def get_actions(state, from_x, from_y):
    piece = state[from_y][from_x]

    side = piece[0]

    action_list = []

    # 대각선 오른쪽 전진 길 체크
    if ((from_y == 9 and from_x == 3) or (from_y == 8 and from_x == 4)) and \
            not kcu.is_our_side(state, from_x + 1, from_y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x + 1, y - 1, side):
        action_list.append(
            {'from_x': from_x, 'from_y': from_y, 'to_x': from_x + 1, 'to_y': from_y - 1})

    # 대각선 왼쪽 전진 길 체크
    if ((from_y == 9 and from_x == 5) or (from_y == 8 and from_x == 4)) and \
            not kcu.is_our_side(state, from_x - 1, from_y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x - 1, y - 1, side):
        action_list.append(
            {'from_x': from_x, 'from_y': from_y, 'to_x': from_x - 1, 'to_y': from_y - 1})

    # 대각선 오른쪽 후진 길 체크
    if ((from_y == 7 and from_x == 3) or (from_y == 8 and from_x == 4)) and \
            not kcu.is_our_side(state, from_x + 1, from_y + 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x + 1, y + 1, side):
        action_list.append(
            {'from_x': from_x, 'from_y': from_y, 'to_x': from_x + 1, 'to_y': from_y + 1})

    # 대각선 왼쪽 후진 길 체크
    if ((from_y == 7 and from_x == 5) or (from_y == 8 and from_x == 4)) and \
            not kcu.is_our_side(state, from_x - 1, from_y + 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x - 1, y + 1, side):
        action_list.append(
            {'from_x': from_x, 'from_y': from_y, 'to_x': from_x - 1, 'to_y': from_y + 1})

    # 전진 길 체크
    if from_y in (8, 9) and not kcu.is_our_side(state, from_x, from_y - 1, side):
        # and   not kcu.is_losing_way(state_map, x, y, x, y - 1, side):
        action_list.append(
            {'from_x': from_x, 'from_y': from_y, 'to_x': from_x, 'to_y': from_y - 1})

    # 오른쪽 길 체크
    if from_x in (3, 4) and not kcu.is_our_side(state, from_x + 1, from_y, side):
        # and  not kcu.is_losing_way(state_map, x, y, x + 1, y, side):
        action_list.append(
            {'from_x': from_x, 'from_y': from_y, 'to_x': from_x + 1, 'to_y': from_y})

    # 왼쪽 길 체크
    if from_x in (4, 5) and not kcu.is_our_side(state, from_x - 1, from_y, side):
        # and   not kcu.is_losing_way(state_map, x, y, x - 1, y, side):
        action_list.append(
            {'from_x': from_x, 'from_y': from_y, 'to_x': from_x - 1, 'to_y': from_y})

    # 후진 길 체크
    if from_y in (7, 8) and not kcu.is_our_side(state, from_x, from_y + 1, side):
        # and   not kcu.is_losing_way(state_map, x, y, x, y + 1, side):
        action_list.append(
            {'from_x': from_x, 'from_y': from_y, 'to_x': from_x, 'to_y': from_y + 1})

    return action_list
