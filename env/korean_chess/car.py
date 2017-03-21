# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env.korean_chess import common


def get_actions(state_map, x, y):
    piece = state_map[y][x]

    side = piece[0]

    action_list = []

    piece_type = common.CAR

    # 대각선 오른쪽 전진 길 체크
    if (y == 9 and x == 3) or (y == 2 and x == 3):
        moving_y = y - 1
        moving_x = x + 1
        step = 1
        while moving_y >= y - 2:
            if not common.is_our_side(state_map, moving_x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, moving_y, side):
                action_list.append(
                    {'x': x, 'y': y, 'to_x': moving_x, 'to_y': moving_y, 'direction': common.DIAGONAL_FORWARD_RIGHT1,
                     'step': step,
                     'piece_type': piece_type})
                if common.is_enemy(state_map, moving_x, moving_y, side):
                    break
            else:
                break
            moving_y -= 1
            moving_x += 1
            step += 1

    if (y == 8 and x == 4) or (y == 1 and x == 4):
        if not common.is_our_side(state_map, x + 1, y - 1, side):
            # if not util.is_losing_way(state_map, x, y, x + 1, y - 1, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x + 1, 'to_y': y - 1, 'direction': common.DIAGONAL_FORWARD_RIGHT1,
                 'step': 1,
                 'piece_type': piece_type})

    # 대각선 왼쪽 전진 길 체크
    if (y == 9 and x == 5) or (y == 2 and x == 5):
        moving_y = y - 1
        moving_x = x - 1
        step = 1
        while moving_y >= y - 2:
            if not common.is_our_side(state_map, moving_x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, moving_y, side):
                action_list.append(
                    {'x': x, 'y': y, 'to_x': moving_x, 'to_y': moving_y, 'direction': common.DIAGONAL_FORWARD_LEFT1,
                     'step': step,
                     'piece_type': piece_type})
                if common.is_enemy(state_map, moving_x, moving_y, side):
                    break
            else:
                break
            moving_y -= 1
            moving_x -= 1
            step += 1

    if (y == 8 and x == 4) or (y == 1 and x == 4):
        if not common.is_our_side(state_map, x - 1, y - 1, side):
            # if not util.is_losing_way(state_map, x, y, x - 1, y - 1, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x - 1, 'to_y': y - 1, 'direction': common.DIAGONAL_FORWARD_LEFT1, 'step': 1,
                 'piece_type': piece_type})

    # 대각선 오른쪽 후진 길 체크
    if (y == 7 and x == 3) or (y == 0 and x == 3):
        moving_y = y + 1
        moving_x = x + 1
        step = 1
        while moving_y <= y + 2:
            if not common.is_our_side(state_map, moving_x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, moving_y, side):
                action_list.append(
                    {'x': x, 'y': y, 'to_x': moving_x, 'to_y': moving_y, 'direction': common.DIAGONAL_BACKWARD_RIGHT1,
                     'step': step,
                     'piece_type': piece_type})
                if common.is_enemy(state_map, moving_x, moving_y, side):
                    break
            else:
                break
            moving_y += 1
            moving_x += 1
            step += 1

    if (y == 8 and x == 4) or (y == 1 and x == 4):
        if not common.is_our_side(state_map, x + 1, y + 1, side):
            # if not util.is_losing_way(state_map, x, y, x + 1, y + 1, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x + 1, 'to_y': y + 1, 'direction': common.DIAGONAL_BACKWARD_RIGHT1,
                 'step': 1,
                 'piece_type': piece_type})

    # 대각선 왼쪽 후진 길 체크
    if (y == 7 and x == 5) or (y == 0 and x == 5):
        moving_y = y + 1
        moving_x = x - 1
        step = 1
        while moving_y <= y + 2:
            if not common.is_our_side(state_map, moving_x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, moving_y, side):
                action_list.append(
                    {'x': x, 'y': y, 'to_x': moving_x, 'to_y': moving_y, 'direction': common.DIAGONAL_BACKWARD_LEFT1,
                     'step': step,
                     'piece_type': piece_type})
                if common.is_enemy(state_map, moving_x, moving_y, side):
                    break
            else:
                break
            moving_y += 1
            moving_x -= 1
            step += 1

    if (y == 8 and x == 4) or (y == 1 and x == 4):
        if not common.is_our_side(state_map, x - 1, y + 1, side):
            # if not util.is_losing_way(state_map, x, y, x - 1, y + 1, side):
            action_list.append(
                {'x': x, 'y': y, 'to_x': x -1, 'to_y': y + 1, 'direction': common.DIAGONAL_BACKWARD_LEFT1,
                 'step': 1,
                 'piece_type': piece_type})

    # 전진 길 체크
    if y > 0:
        moving_y = y - 1
        step = 1
        while moving_y >= 0:
            if not common.is_our_side(state_map, x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, x, moving_y, side):
                action_list.append(
                    {'x': x, 'y': y, 'to_x': x, 'to_y': moving_y, 'direction': common.FORWARD, 'step': step,
                     'piece_type': piece_type})
                if common.is_enemy(state_map, x, moving_y, side):
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
            if not common.is_our_side(state_map, moving_x, y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, y, side):
                action_list.append(
                    {'x': x, 'y': y, 'to_x': moving_x, 'to_y': y, 'direction': common.RIGHT, 'step': step,
                     'piece_type': piece_type})
                if common.is_enemy(state_map, moving_x, y, side):
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
            if not common.is_our_side(state_map, moving_x, y, side):
                # if not util.is_losing_way(state_map, x, y, moving_x, y, side):
                action_list.append(
                    {'x': x, 'y': y, 'to_x': moving_x, 'to_y': y, 'direction': common.LEFT, 'step': step,
                     'piece_type': piece_type})
                if common.is_enemy(state_map, moving_x, y, side):
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
            if not common.is_our_side(state_map, x, moving_y, side):
                # if not util.is_losing_way(state_map, x, y, x, moving_y, side):
                action_list.append(
                    {'x': x, 'y': y, 'to_x': x, 'to_y': moving_y, 'direction': common.BACKWARD, 'step': step,
                     'piece_type': piece_type})

                if common.is_enemy(state_map, x, moving_y, side):
                    break
            else:
                break
            moving_y += 1
            step += 1

    return action_list
