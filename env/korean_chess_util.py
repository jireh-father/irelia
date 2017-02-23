from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def is_empty_space(state_map, x, y):
    return state_map[y][x] is 0


def is_our_side(state_map, x, y, side):
    return state_map[y][x] != 0 and state_map[y][x][0] == side


def is_enemy(state_map, x, y, side):
    return state_map[y][x] != 0 and state_map[y][x][0] != side


def is_losing_way(state_map, x, y, to_x, to_y, side):
    return False


def is_cannon(state_map, x, y):
    return state_map[y][x] is not 0 and int(state_map[y][x][1]) is CANNON


def is_piece(state_map, x, y):
    return state_map[y][x] is not 0
