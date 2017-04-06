from core.constant import Constant


def add_action_to_list(x, y, to_x, to_y, list, color, is_hash_map):
    if color is Constant.RED:
        x = 8 - x
        y = 9 - y
        to_x = 8 - to_x
        to_y = 9 - to_y
    action = {'x': x, 'y': y, 'to_x': to_x, 'to_y': to_y}
    if is_hash_map:
        list["%d_%d_%d_%d" % (x, y, to_x, to_y)] = True  # action
    else:
        list.append(action)


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
