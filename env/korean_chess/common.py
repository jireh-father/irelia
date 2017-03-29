from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

KING = 7
SOLDIER = 1
SANG = 2
GUARDIAN = 3
HORSE = 4
CANNON = 5
CAR = 6

BLUE = 'b'
RED = 'r'

FORWARD = 0
RIGHT = 1
LEFT = 2
BACKWARD = 3
DIAGONAL_FORWARD_RIGHT1 = 4
DIAGONAL_FORWARD_RIGHT2 = 5
DIAGONAL_FORWARD_LEFT1 = 6
DIAGONAL_FORWARD_LEFT2 = 7
DIAGONAL_BACKWARD_RIGHT1 = 8
DIAGONAL_BACKWARD_RIGHT2 = 9
DIAGONAL_BACKWARD_LEFT1 = 10
DIAGONAL_BACKWARD_LEFT2 = 11

LEFT_WALL = 0
RIGHT_WALL = 8
TOP_WALL = 0
BOTTOM_WALL = 9

CONV_PIECE_LIST = {
    SOLDIER: '0.15',
    SANG: '0.29',
    GUARDIAN: '0.43',
    HORSE: '0.58',
    CANNON: '0.72',
    CAR: '0.86',
    KING: '1.0',
}


def is_empty_space(state_map, x, y):
    return state_map[y][x] is 0


def is_our_side(state_map, x, y, side):
    return state_map[y][x] != 0 and state_map[y][x][0] == side


def is_enemy(state_map, x, y, side):
    return state_map[y][x] != 0 and state_map[y][x][0] != side


def is_checkmate_try(state_map, to_x, to_y, side):
    return state_map[to_y][to_x][1] == KING and state_map[to_y][to_x][0] != side


def is_checkmate(state_map, x, y, to_x, to_y, side):
    return False


def is_stalemate():
    return False


def is_cannon(state_map, x, y):
    return state_map[y][x] is not 0 and int(state_map[y][x][1]) is CANNON


def is_piece(state_map, x, y):
    return state_map[y][x] is not 0


def convert_state_feature_map(state_key, color='b'):
    blue_feature_map = []
    red_feature_map = []
    for piece in state_key.split(','):
        if piece.isdigit():
            blue_feature_map += ['0'] * int(piece)
            red_feature_map += ['0'] * int(piece)
        else:
            if piece[0] is 'b':
                blue_feature_map.append(CONV_PIECE_LIST[int(piece[1])])
                red_feature_map.append('0')
            else:
                red_feature_map.append(CONV_PIECE_LIST[int(piece[1])])
                blue_feature_map.append('0')
    if color is 'b':
        color_feature_map = ['0.5'] * 10 * 9
    else:
        color_feature_map = ['1'] * 10 * 9

    result = np.array(blue_feature_map + red_feature_map + color_feature_map)
    result = result.astype(np.float16)
    return result.reshape(-1, 3, 10, 9)
