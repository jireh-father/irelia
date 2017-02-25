# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env import korean_chess_util as kcu


def get_actions(state_map, x, y):
    # todo: 구현
    piece = state_map[y][x]

    side = piece[0]

    action_list = []

    piece_type = kcu.HORSE

    # 대각선 오른쪽 전진 1 길 체크
    if x < kcu.RIGHT_WALL and y > kcu.TOP_WALL + 1:
        if kcu.is_empty_space(state_map, x, y - 1) and not kcu.is_our_side(x + 1, y - 2):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x + 1, 'to_y': y - 2, 'direction': kcu.DIAGONAL_FORWARD_RIGHT1, 'step': 1,
                 'piece_type': piece_type})

    # 대각선 오른쪽 전진 2 길 체크

    # 대각선 왼쪽 전진 1 길 체크

    # 대각선 왼쪽 전진 2 길 체크

    # 대각선 오른쪽 후진 1 길 체크

    # 대각선 오른쪽 후진 2 길 체크

    # 대각선 왼쪽 후진 1 길 체크

    # 대각선 왼쪽 후진 2 길 체크

    # [공통]
    # 우리편있는지 체크
    # 장군 검증

    return []
