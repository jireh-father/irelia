# coding=utf8
from core.constant import Constant
from core.piece import common


def get_actions(state_map, x, y, color=Constant.BLUE, is_hash_map=False):
    piece = state_map[y][x]
    side = piece[0]

    if is_hash_map:
        action_list = {}
    else:
        action_list = []

    # 대각선 오른쪽 길 체크
    if ((y == 1 and x == 4) or (y == 2 and x == 3)) and \
      not common.is_our_side(state_map, x + 1, y - 1, side):
        # and not util.is_losing_way(state_map, x, y, x + 1, y - 1, side):
        common.add_action_to_list(x, y, x + 1, y - 1, action_list, color, is_hash_map)

    # 대각선 왼쪽 길 체크
    if ((y == 1 and x == 4) or (y == 2 and x == 5)) and \
      not common.is_our_side(state_map, x - 1, y - 1, side):
        # and not util.is_losing_way(state_map, x, y, x - 1, y - 1, side):
        common.add_action_to_list(x, y, x - 1, y - 1, action_list, color, is_hash_map)

    # 전진 길 체크
    if y != 0 and not common.is_our_side(state_map, x, y - 1, side):
        # and not util.is_losing_way(state_map, x, y, x, y - 1, side):
        common.add_action_to_list(x, y, x, y - 1, action_list, color, is_hash_map)

    # 오른쪽 길 체크
    if x != 8 and not common.is_our_side(state_map, x + 1, y, side):
        # and   not util.is_losing_way(state_map, x, y, x + 1, y, side):
        common.add_action_to_list(x, y, x + 1, y, action_list, color, is_hash_map)

    # 왼쪽 길 체크
    if x != 0 and not common.is_our_side(state_map, x - 1, y, side):
        # and   not util.is_losing_way(state_map, x, y, x - 1, y, side):
        common.add_action_to_list(x, y, x - 1, y, action_list, color, is_hash_map)

    return action_list
