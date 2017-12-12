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

LEFT_WALL = 0
RIGHT_WALL = 8
TOP_WALL = 0
BOTTOM_WALL = 9

REWARD_LIST = {
    SOLDIER: 2,
    SANG: 3,
    GUARDIAN: 3,
    HORSE: 5,
    CANNON: 7,
    CAR: 13,
    KING: 73,
}


def get_score(state, turn):
    print("test", state)
    score = 0
    for line in state:
        for piece_num in line:
            if piece_num == 0 or piece_num[0] != turn or int(piece_num[1]) == KING:
                continue
            score += REWARD_LIST[int(piece_num[1])]
    if turn == RED:
        score += 1.5
    return score


def is_empty_space(state, x, y):
    return state[y][x] == 0


def is_our_side(state, x, y, side):
    return state[y][x] != 0 and state[y][x][0] == side


def is_enemy(state, x, y, side):
    return state[y][x] != 0 and state[y][x][0] != side


def is_cannon(state, x, y):
    return state[y][x] != 0 and int(state[y][x][1]) == CANNON


def is_piece(state, x, y):
    return state[y][x] != 0
