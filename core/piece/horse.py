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

    # 대각선 오른쪽 전진 1 길 체크
    if x < Constant.RIGHT_WALL and y > Constant.TOP_WALL + 1:
        if common.is_empty_space(state_map, x, y - 1) and not common.is_our_side(state_map, x + 1, y - 2, side):
            common.add_action_to_list(x, y, x + 1, y - 2, action_list, color, is_hash_map)

    # 대각선 오른쪽 전진 2 길 체크
    if x < Constant.RIGHT_WALL - 1 and y > Constant.TOP_WALL:
        if common.is_empty_space(state_map, x + 1, y) and not common.is_our_side(state_map, x + 2, y - 1, side):
            common.add_action_to_list(x, y, x + 2, y - 1, action_list, color, is_hash_map)

    # 대각선 왼쪽 전진 1 길 체크
    if x > Constant.LEFT_WALL and y > Constant.TOP_WALL + 1:
        if common.is_empty_space(state_map, x, y - 1) and not common.is_our_side(state_map, x - 1, y - 2, side):
            common.add_action_to_list(x, y, x - 1, y - 2, action_list, color, is_hash_map)

    # 대각선 왼쪽 전진 2 길 체크
    if x > Constant.LEFT_WALL + 1 and y > Constant.TOP_WALL:
        if common.is_empty_space(state_map, x - 1, y) and not common.is_our_side(state_map, x - 2, y - 1, side):
            common.add_action_to_list(x, y, x - 2, y - 1, action_list, color, is_hash_map)

    # 대각선 오른쪽 후진 1 길 체크
    if x < Constant.RIGHT_WALL and y < Constant.BOTTOM_WALL - 1:
        if common.is_empty_space(state_map, x, y + 1) and not common.is_our_side(state_map, x + 1, y + 2, side):
            common.add_action_to_list(x, y, x + 1, y + 2, action_list, color, is_hash_map)

    # 대각선 오른쪽 후진 2 길 체크
    if x < Constant.RIGHT_WALL - 1 and y < Constant.BOTTOM_WALL:
        if common.is_empty_space(state_map, x + 1, y) and not common.is_our_side(state_map, x + 2, y + 1, side):
            common.add_action_to_list(x, y, x + 2, y + 1, action_list, color, is_hash_map)

    # 대각선 왼쪽 후진 1 길 체크
    if x > Constant.LEFT_WALL and y < Constant.BOTTOM_WALL - 1:
        if common.is_empty_space(state_map, x, y + 1) and not common.is_our_side(state_map, x - 1, y + 2, side):
            common.add_action_to_list(x, y, x - 1, y + 2, action_list, color, is_hash_map)

    # 대각선 왼쪽 후진 2 길 체크
    if x > Constant.LEFT_WALL + 1 and y < Constant.BOTTOM_WALL:
        if common.is_empty_space(state_map, x - 1, y) and not common.is_our_side(state_map, x - 2, y + 1, side):
            common.add_action_to_list(x, y, x - 2, y + 1, action_list, color, is_hash_map)

    return action_list
