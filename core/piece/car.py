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
    if (y == 9 and x == 3) or (y == 2 and x == 3):
        moving_y = y - 1
        moving_x = x + 1
        step = 1
        while moving_y >= y - 2:
            if not game.is_our_side(state_map, moving_x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, moving_y, side):
                common.add_action_to_list(x, y, moving_x, moving_y, action_list, color, is_hash_map)
                if game.is_enemy(state_map, moving_x, moving_y, side):
                    break
            else:
                break
            moving_y -= 1
            moving_x += 1
            step += 1

    if (y == 8 and x == 4) or (y == 1 and x == 4):
        if not game.is_our_side(state_map, x + 1, y - 1, side):
            # if not util.is_losing_way(state_map, x, y, x + 1, y - 1, side):
            common.add_action_to_list(x, y, x + 1, y - 1, action_list, color, is_hash_map)

    # 대각선 왼쪽 전진 길 체크
    if (y == 9 and x == 5) or (y == 2 and x == 5):
        moving_y = y - 1
        moving_x = x - 1
        step = 1
        while moving_y >= y - 2:
            if not game.is_our_side(state_map, moving_x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, moving_y, side):
                common.add_action_to_list(x, y, moving_x, moving_y, action_list, color, is_hash_map)
                if game.is_enemy(state_map, moving_x, moving_y, side):
                    break
            else:
                break
            moving_y -= 1
            moving_x -= 1
            step += 1

    if (y == 8 and x == 4) or (y == 1 and x == 4):
        if not game.is_our_side(state_map, x - 1, y - 1, side):
            # if not util.is_losing_way(state_map, x, y, x - 1, y - 1, side):
            common.add_action_to_list(x, y, x - 1, y - 1, action_list, color, is_hash_map)

    # 대각선 오른쪽 후진 길 체크
    if (y == 7 and x == 3) or (y == 0 and x == 3):
        moving_y = y + 1
        moving_x = x + 1
        step = 1
        while moving_y <= y + 2:
            if not game.is_our_side(state_map, moving_x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, moving_y, side):
                common.add_action_to_list(x, y, moving_x, moving_y, action_list, color, is_hash_map)
                if game.is_enemy(state_map, moving_x, moving_y, side):
                    break
            else:
                break
            moving_y += 1
            moving_x += 1
            step += 1

    if (y == 8 and x == 4) or (y == 1 and x == 4):
        if not game.is_our_side(state_map, x + 1, y + 1, side):
            # if not util.is_losing_way(state_map, x, y, x + 1, y + 1, side):
            common.add_action_to_list(x, y, x + 1, y + 1, action_list, color, is_hash_map)

    # 대각선 왼쪽 후진 길 체크
    if (y == 7 and x == 5) or (y == 0 and x == 5):
        moving_y = y + 1
        moving_x = x - 1
        step = 1
        while moving_y <= y + 2:
            if not game.is_our_side(state_map, moving_x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, moving_y, side):
                common.add_action_to_list(x, y, moving_x, moving_y, action_list, color, is_hash_map)
                if game.is_enemy(state_map, moving_x, moving_y, side):
                    break
            else:
                break
            moving_y += 1
            moving_x -= 1
            step += 1

    if (y == 8 and x == 4) or (y == 1 and x == 4):
        if not game.is_our_side(state_map, x - 1, y + 1, side):
            # if not util.is_losing_way(state_map, x, y, x - 1, y + 1, side):
            common.add_action_to_list(x, y, x - 1, y + 1, action_list, color, is_hash_map)

    # 전진 길 체크
    if y > 0:
        moving_y = y - 1
        step = 1
        while moving_y >= 0:
            if not game.is_our_side(state_map, x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, x, moving_y, side):
                common.add_action_to_list(x, y, x, moving_y, action_list, color, is_hash_map)
                if game.is_enemy(state_map, x, moving_y, side):
                    break
            else:
                break
            moving_y -= 1
            step += 1

    # 오른쪽 길 체크
    if x < 8:
        moving_x = x + 1
        step = 1
        while moving_x <= 8:
            if not game.is_our_side(state_map, moving_x, y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, y, side):
                common.add_action_to_list(x, y, moving_x, y, action_list, color, is_hash_map)
                if game.is_enemy(state_map, moving_x, y, side):
                    break
            else:
                break
            moving_x += 1
            step += 1

    # 왼쪽 길 체크
    if x > 0:
        moving_x = x - 1
        step = 1
        while moving_x >= 0:
            if not game.is_our_side(state_map, moving_x, y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, y, side):
                common.add_action_to_list(x, y, moving_x, y, action_list, color, is_hash_map)
                if game.is_enemy(state_map, moving_x, y, side):
                    break
            else:
                break
            moving_x -= 1
            step += 1

    # 후진 길 체크
    if y < 9:
        moving_y = y + 1
        step = 1
        while moving_y <= 9:
            if not game.is_our_side(state_map, x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, x, moving_y, side):
                common.add_action_to_list(x, y, x, moving_y, action_list, color, is_hash_map)

                if game.is_enemy(state_map, x, moving_y, side):
                    break
            else:
                break
            moving_y += 1
            step += 1

    return action_list
