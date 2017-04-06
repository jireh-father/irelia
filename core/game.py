from core.piece import piece_factory
from core.constant import Constant
import numpy as np


def get_actions_hash_map(state_key, color):
    action_list = {}
    if color is Constant.RED:
        state_key = reverse_state_key(state_key)
    state_map = convert_state_map(state_key)
    for y, line in enumerate(state_map):
        for x, piece in enumerate(line):
            if piece == 0 or piece[0] != color:
                continue
            action_list.update(piece_factory.get_actions(state_map, x, y, color, True))

    return action_list


def convert_state_map(state_key):
    state_map = []
    for piece in state_key.split(','):
        if piece.isdigit():
            state_map += [0] * int(piece)
        else:
            state_map.append(piece)
    result = np.array(state_map).reshape([-1, 9]).tolist()
    for i, row in enumerate(result):
        for j, piece in enumerate(row):
            if piece == '0':
                result[i][j] = 0
    return result


def is_empty_space(state_map, x, y):
    return state_map[y][x] is 0


def is_our_side(state_map, x, y, side):
    return state_map[y][x] != 0 and state_map[y][x][0] == side


def is_enemy(state_map, x, y, side):
    return state_map[y][x] != 0 and state_map[y][x][0] != side


def is_checkmate_try(state_map, to_x, to_y, side):
    return state_map[to_y][to_x][1] == Constant.KING and state_map[to_y][to_x][0] != side


def is_checkmate(state_map, x, y, to_x, to_y, side):
    return False


def is_stalemate():
    return False


def is_cannon(state_map, x, y):
    return state_map[y][x] is not 0 and int(state_map[y][x][1]) is Constant.CANNON


def is_piece(state_map, x, y):
    return state_map[y][x] is not 0


def reverse_state_key(state):
    return ','.join(list(reversed(state.split(','))))


def convert_state_feature_map(state_key, color='b'):
    blue_feature_map = []
    red_feature_map = []
    for piece in state_key.split(','):
        if piece.isdigit():
            blue_feature_map += ['0'] * int(piece)
            red_feature_map += ['0'] * int(piece)
        else:
            if piece[0] is 'b':
                blue_feature_map.append(Constant.CONV_PIECE_LIST[int(piece[1])])
                red_feature_map.append('0')
            else:
                red_feature_map.append(Constant.CONV_PIECE_LIST[int(piece[1])])
                blue_feature_map.append('0')
    if color is 'b':
        color_feature_map = ['0.5'] * 10 * 9
    else:
        color_feature_map = ['1'] * 10 * 9

    result = np.array(blue_feature_map + red_feature_map + color_feature_map)
    result = result.astype(np.float16)
    return result.reshape(-1, 3, 10, 9)


def convert_one_dim_pos_to_two_dim_pos(position):
    return position % 9, position // 9


def build_pos_key(x, y, to_x, to_y):
    return "%d_%d_%d_%d" % (x, y, to_x, to_y)
