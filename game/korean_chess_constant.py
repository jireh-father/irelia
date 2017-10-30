# coding=utf8

KING = 7
SOLDIER = 1
SANG = 2
GUARDIAN = 3
HORSE = 4
CANNON = 5
CAR = 6

# king 궁(장) gung
B_KG = 'b7'
# soldier 졸(병) jol
B_SD = 'b1'
# sang 상
B_SG = 'b2'
# guardian 사 sa
B_GD = 'b3'
# horse 마 ma
B_HS = 'b4'
# cannon 포 po
B_CN = 'b5'
# car 차 cha
B_CR = 'b6'

R_KG = 'r7'
R_SD = 'r1'
R_SG = 'r2'
R_GD = 'r3'
R_HS = 'r4'
R_CN = 'r5'
R_CR = 'r6'

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

TO_X_IDX = 0
TO_Y_IDX = 1
FROM_X_IDX = 2
FROM_Y_IDX = 3

def is_empty_space(state_map, x, y):
    return state_map[y][x] == 0


def is_our_side(state_map, x, y, side):
    return state_map[y][x] != 0 and state_map[y][x][0] == side


def is_enemy(state_map, x, y, side):
    return state_map[y][x] != 0 and state_map[y][x][0] != side


def is_checkmate_try(state_map, to_x, to_y, side):
    return state_map[to_y][to_x][1] == KING and state_map[to_y][to_x][0] != side


def is_stalemate():
    return False


def is_cannon(state_map, x, y):
    return state_map[y][x] != 0 and int(state_map[y][x][1]) == CANNON


def is_piece(state_map, x, y):
    return state_map[y][x] != 0
