# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env.korean_chess import common


def get_actions(state_map, x, y):
    piece = state_map[y][x]

    side = piece[0]

    action_list = []

    piece_type = common.HORSE

    # 대각선 오른쪽 전진 1 길 체크
    if x < common.RIGHT_WALL and y > common.TOP_WALL + 1:
        if common.is_empty_space(state_map, x, y - 1) and not common.is_our_side(state_map, x + 1, y - 2, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x + 1, 'to_y': y - 2, 'direction': common.DIAGONAL_FORWARD_RIGHT1, 'step': 1,
                 'piece_type': piece_type})

    # 대각선 오른쪽 전진 2 길 체크
    if x < common.RIGHT_WALL - 1 and y > common.TOP_WALL:
        if common.is_empty_space(state_map, x + 1, y) and not common.is_our_side(state_map, x + 2, y - 1, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x + 2, 'to_y': y - 1, 'direction': common.DIAGONAL_FORWARD_RIGHT2, 'step': 1,
                 'piece_type': piece_type})

    # 대각선 왼쪽 전진 1 길 체크
    if x > common.LEFT_WALL and y > common.TOP_WALL + 1:
        if common.is_empty_space(state_map, x, y - 1) and not common.is_our_side(state_map, x - 1, y - 2, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x - 1, 'to_y': y - 2, 'direction': common.DIAGONAL_FORWARD_LEFT1, 'step': 1,
                 'piece_type': piece_type})

    # 대각선 왼쪽 전진 2 길 체크
    if x > common.LEFT_WALL + 1 and y > common.TOP_WALL:
        if common.is_empty_space(state_map, x - 1, y) and not common.is_our_side(state_map, x - 2, y - 1, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x - 2, 'to_y': y - 1, 'direction': common.DIAGONAL_FORWARD_LEFT2
                    , 'step': 1,
                 'piece_type': piece_type})

    # 대각선 오른쪽 후진 1 길 체크
    if x < common.RIGHT_WALL and y < common.BOTTOM_WALL - 1:
        if common.is_empty_space(state_map, x, y + 1) and not common.is_our_side(state_map, x + 1, y + 2, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x + 1, 'to_y': y + 2, 'direction': common.DIAGONAL_BACKWARD_RIGHT1, 'step': 1,
                 'piece_type': piece_type})

    # 대각선 오른쪽 후진 2 길 체크
    if x < common.RIGHT_WALL - 1 and y < common.BOTTOM_WALL:
        if common.is_empty_space(state_map, x + 1, y) and not common.is_our_side(state_map, x + 2, y + 1, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x + 2, 'to_y': y + 1, 'direction': common.DIAGONAL_BACKWARD_RIGHT2, 'step': 1,
                 'piece_type': piece_type})

    # 대각선 왼쪽 후진 1 길 체크
    if x > common.LEFT_WALL and y < common.BOTTOM_WALL - 1:
        if common.is_empty_space(state_map, x, y + 1) and not common.is_our_side(state_map, x - 1, y + 2, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x - 1, 'to_y': y + 2, 'direction': common.DIAGONAL_BACKWARD_LEFT1, 'step': 1,
                 'piece_type': piece_type})

    # 대각선 왼쪽 후진 2 길 체크
    if x > common.LEFT_WALL + 1 and y < common.BOTTOM_WALL:
        if common.is_empty_space(state_map, x - 1, y) and not common.is_our_side(state_map, x - 2, y + 1, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x - 2, 'to_y': y + 1, 'direction': common.DIAGONAL_BACKWARD_LEFT2, 'step': 1,
                 'piece_type': piece_type})

    return action_list
