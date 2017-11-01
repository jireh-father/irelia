# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from game.korean_chess_piece import piece_factory
from game import korean_chess_constant as c


def reverse_state(state, is_copy=True):
    if is_copy:
        state = copy_state(state)
    state.reverse()
    reversed_state = []
    for line in state:
        line.reverse()
        reversed_state.append(line)
    return reversed_state


def reverse_action(from_x, from_y, to_x, to_y):
    return 8 - from_x, 9 - from_y, 8 - to_x, 9 - to_y


def copy_state(state):
    return copy.deepcopy(state)


def validate_action(action, state, turn, next_turn):
    to_x = action['to_x']
    to_y = action['to_y']
    from_x = action['from_x']
    from_y = action['from_y']

    # check the piece is empty
    piece_num = int(state[from_y][from_x][-1])
    if piece_num == 0:
        return False

    # check the piece is current turn.
    if state[from_y][from_x][0] != turn:
        return False

    actions = get_actions(state, from_x, from_y, turn)

    if not actions:
        return False

    # check there is a valid action.
    invalid_cnt = 0
    for tmp_action in actions:
        if tmp_action["to_x"] != to_x or tmp_action["to_y"] != to_y:
            invalid_cnt += 1

    if invalid_cnt == len(actions):
        return False

    # check this action gets my own checkmate.
    return not is_check(current_state, from_x, from_y, to_x, to_y, next_turn)


def decode_state(state, turn):
    new_state = [[[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9],
                 [[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9]]
    for i, line in enumerate(state):
        for j, piece in enumerate(line):
            if piece != 0:
                if piece[0] == c.BLUE:
                    new_state[0][i][j] = piece[1]
                else:
                    new_state[1][i][j] = piece[1]
    if turn == c.BLUE:
        new_state.append([[1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9])
    else:
        new_state.append([[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9])

    return new_state


def encode_state(state):
    new_state = [[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9]
    for i, line in enumerate(state[0]):
        for j, piece in enumerate(line):
            if piece != 0:
                new_state[i][j] = "b" + str(piece)
            if state[1][i][j] != 0:
                new_state[i][j] = "r" + str(state[1][i][j])
    turn = c.BLUE if state[2][0][0] == 1 else c.RED
    return new_state, turn


def is_check(state, from_x, from_y, to_x, to_y, turn):
    state = copy_state(state)
    state[to_y][to_x] = state[from_y][from_x]
    state[from_y][from_x] = 0
    state = reverse_state(state)
    for y, line in enumerate(state):
        for x, piece_num in enumerate(line):
            if piece_num == 0 or piece_num[0] != turn:
                continue
            piece = piece_factory.get_piece(int(piece_num[1]))

            actions = piece.get_actions(state, x, y)
            for action in actions:
                if state[action["to_y"]][action["to_x"]] != 0 \
                        and int(state[action["to_y"]][action["to_x"]][1]) == c.KING \
                        and state[action["to_y"]][action["to_x"]][0] != turn:
                    return True
    return False


def is_checkmate(state, turn):
    if turn == c.BLUE:
        r_state = reverse_state(state, False)
        subject_turn = c.RED
    else:
        r_state = copy_state(state)
        subject_turn = c.BLUE
    subject_actions = []
    for y, line in enumerate(r_state):
        for x, piece_num in enumerate(line):
            if piece_num != 0 and piece_num[0] != turn:
                piece = piece_factory.get_piece(int(piece_num[1]))
                subject_actions += piece.get_actions(r_state, x, y, )
    state = reverse_state(r_state, False)
    for subject_action in subject_actions:
        to_x = subject_action['to_x']
        to_y = subject_action['to_y']
        from_x = subject_action['from_x']
        from_y = subject_action['from_y']
        old = state[to_y][to_x]
        state[to_y][to_x] = state[from_y][from_x]
        state[from_y][from_x] = 0
        for y, line in enumerate(state):
            for x, piece_num in enumerate(line):
                if piece_num != 0 and piece_num[0] == turn:
                    piece = piece_factory.get_piece(int(piece_num[1]))
                    actions = piece.get_actions(state, x, y)
                    check_cnt = 0
                    for action in actions:
                        if is_check(state, action["from_x"], action["from_y"], action["to_x"], action["to_y"], turn):
                            check_cnt += 1
                    if check_cnt == 0:
                        return False

        state[from_y][from_x] = state[to_y][to_x]
        state[to_y][to_x] = old

    return True


def is_draw(state):
    for line in state:
        for piece in line:
            if piece == 0:
                continue
            # todo: 상하고 마하고 구현해면 빼
            if piece[1] != c.KING and piece[1] != c.GUARDIAN:
                return False
                # todo: 포만 남았을경우 못넘는경우면 비긴걸로 계산
    return True


def get_all_actions(state, turn):
    if turn == c.RED:
        state = reverse_state(state)
    action_list = []
    for y, line in enumerate(state):
        for x, piece_num in enumerate(line):
            if piece_num == 0 or piece_num[0] != turn:
                continue

            piece = piece_factory.get_piece(int(piece_num[1]))
            action_list += [reverse_action() for action in piece.get_actions(state, x, y)]
    return action_list


def get_actions(state, x, y, turn):
    if turn == c.RED:
        state = reverse_state(state)
    piece = piece_factory.get_piece(int(state[y][x][1]))

    return piece.get_actions(state, x, y)
