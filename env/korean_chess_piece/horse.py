# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env import korean_chess_util


def get_actions(state_map, x, y):
    # todo: 구현
    piece = state_map[y][x]

    side = piece[0]

    action_list = []

    piece_type = korean_chess_util.HORSE

    # 대각선 오른쪽 전진 1 길 체크

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
