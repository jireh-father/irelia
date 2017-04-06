# coding=utf8

from core.constant import Constant
from core import game
from core.piece import common


def get_actions(state_map, x, y, color=Constant.BLUE, is_hash_map=False):
    piece = state_map[y][x]

    side = piece[0]

    if is_hash_map:
        action_list = {}
    else:
        action_list = []

    # 대각선 오른쪽 전진 길 체크
    if ((y == 9 and x == 3) or (y == 8 and x == 4)) and \
      not game.is_our_side(state_map, x + 1, y - 1, side):
        # and not util.is_losing_way(state_map, x, y, x + 1, y - 1, side):
        common.add_action_to_list(x, y, x + 1, y - 1, action_list, color, is_hash_map)

    # 대각선 왼쪽 전진 길 체크
    if ((y == 9 and x == 5) or (y == 8 and x == 4)) and \
      not game.is_our_side(state_map, x - 1, y - 1, side):
        # and not util.is_losing_way(state_map, x, y, x - 1, y - 1, side):
        common.add_action_to_list(x, y, x - 1, y - 1, action_list, color, is_hash_map)

    # 대각선 오른쪽 후진 길 체크
    if ((y == 7 and x == 3) or (y == 8 and x == 4)) and \
      not game.is_our_side(state_map, x + 1, y + 1, side):
        # and not util.is_losing_way(state_map, x, y, x + 1, y + 1, side):
        common.add_action_to_list(x, y, x + 1, y + 1, action_list, color, is_hash_map)

    # 대각선 왼쪽 후진 길 체크
    if ((y == 7 and x == 5) or (y == 8 and x == 4)) and \
      not game.is_our_side(state_map, x - 1, y + 1, side):
        # and not util.is_losing_way(state_map, x, y, x - 1, y + 1, side):
        common.add_action_to_list(x, y, x - 1, y + 1, action_list, color, is_hash_map)

    # 전진 길 체크
    if y in (8, 9) and not game.is_our_side(state_map, x, y - 1, side):
        # and   not util.is_losing_way(state_map, x, y, x, y - 1, side):
        common.add_action_to_list(x, y, x, y - 1, action_list, color, is_hash_map)

    # 오른쪽 길 체크
    if x in (3, 4) and not game.is_our_side(state_map, x + 1, y, side):
        # and  not util.is_losing_way(state_map, x, y, x + 1, y, side):
        common.add_action_to_list(x, y, x + 1, y, action_list, color, is_hash_map)

    # 왼쪽 길 체크
    if x in (4, 5) and not game.is_our_side(state_map, x - 1, y, side):
        # and   not util.is_losing_way(state_map, x, y, x - 1, y, side):
        common.add_action_to_list(x, y, x - 1, y, action_list, color, is_hash_map)

    # 후진 길 체크
    if y in (7, 8) and not game.is_our_side(state_map, x, y + 1, side):
        # and   not util.is_losing_way(state_map, x, y, x, y + 1, side):
        common.add_action_to_list(x, y, x, y + 1, action_list, color, is_hash_map)

    return action_list
