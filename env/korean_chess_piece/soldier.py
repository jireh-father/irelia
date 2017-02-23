from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env import korean_chess_util as kcu


def get_actions(state_map, x, y):
    piece = state_map[y][x]
    side = piece[0]

    action_list = []

    piece_type = kcu.SOLDIER

    # 대각선 오른쪽 길 체크
    if ((y == 1 and x == 4) or (y == 2 and x == 3)) and \
      not kcu.is_our_side(state_map, x + 1, y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x + 1, y - 1, side):
        action_list.append({'x': x, 'y': y, 'direction': kcu.DIAGONAL_FORWARD_RIGHT1, 'step': 1,
                            'piece_type': piece_type})

    # 대각선 왼쪽 길 체크
    if ((y == 1 and x == 4) or (y == 2 and x == 5)) and \
      not kcu.is_our_side(state_map, x - 1, y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x - 1, y - 1, side):
        action_list.append({'x': x, 'y': y, 'direction': kcu.DIAGONAL_FORWARD_LEFT1, 'step': 1,
                            'piece_type': piece_type})

    # 전진 길 체크
    if y != 0 and not kcu.is_our_side(state_map, x, y - 1, side):
        # and not kcu.is_losing_way(state_map, x, y, x, y - 1, side):
        action_list.append(
            {'x': x, 'y': y, 'direction': kcu.FORWARD, 'step': 1, 'piece_type': piece_type})

    # 오른쪽 길 체크
    if x != 8 and not kcu.is_our_side(state_map, x + 1, y, side):
        # and   not kcu.is_losing_way(state_map, x, y, x + 1, y, side):
        action_list.append({'x': x, 'y': y, 'direction': kcu.RIGHT, 'step': 1, 'piece_type': piece_type})

    # 왼쪽 길 체크
    if x != 0 and not kcu.is_our_side(state_map, x - 1, y, side):
        # and   not kcu.is_losing_way(state_map, x, y, x - 1, y, side):
        action_list.append({'x': x, 'y': y, 'direction': kcu.LEFT, 'step': 1, 'piece_type': piece_type})

    return action_list
