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


def copy_state(state):
    return copy.deepcopy(state)


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


def validate_action(action, state, turn, next_turn):
    to_x = action['to_x']
    to_y = action['to_y']
    from_x = action['from_x']
    from_y = action['from_y']

    # check the piece is empty
    if state[from_y][from_x] == 0:
        raise Exception("this piece is empty")
        # return False

    # check the piece is current turn.
    if state[from_y][from_x][0] != turn:
        raise Exception("this piece is a opponent piece.")
        # return False

    actions = get_actions(state, from_x, from_y, turn)

    if not actions:
        raise Exception("this piece has no any actions.")
        # return False

    # check there is a valid action.
    invalid_cnt = 0
    for tmp_action in actions:
        if tmp_action["to_x"] != to_x or tmp_action["to_y"] != to_y:
            invalid_cnt += 1

    if invalid_cnt == len(actions):
        raise Exception("this action differs from any actions.")
        # return False

    # check this action gets my own check.
    check = is_check(state, from_x, from_y, to_x, to_y, next_turn)
    if check:
        raise Exception("this action causes opponent's check")
        # return False
    return True


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
    for y, line in enumerate(state):
        for x, piece_num in enumerate(line):
            if piece_num == 0 or piece_num[0] != turn:
                continue
            actions = get_actions(state, x, y, turn)
            for action in actions:
                if state[action["to_y"]][action["to_x"]] != 0 \
                  and int(state[action["to_y"]][action["to_x"]][1]) == c.KING \
                  and state[action["to_y"]][action["to_x"]][0] != turn:
                    return True
    return False


def is_checkmate(state, turn):
    opponent_turn = c.RED if turn == c.BLUE else c.BLUE
    # get opponent actions
    opponent_actions = get_all_actions(state, opponent_turn)
    state = copy_state(state)
    # check opponent's defending move
    for opponent_action in opponent_actions:
        to_x = opponent_action['to_x']
        to_y = opponent_action['to_y']
        from_x = opponent_action['from_x']
        from_y = opponent_action['from_y']
        # move opponent
        old = state[to_y][to_x]
        state[to_y][to_x] = state[from_y][from_x]
        state[from_y][from_x] = 0
        # get my actions after opponent's moving
        next_my_actions = get_all_actions(state, turn)
        # count my check
        check_cnt = 0
        for action in next_my_actions:
            if is_check(state, action["from_x"], action["from_y"], action["to_x"], action["to_y"], turn):
                check_cnt += 1
        if check_cnt == 0:
            return False
        # get back to previous state
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


def reverse_actions(actions):
    for action in actions:
        action["from_x"] = 8 - action["from_x"]
        action["from_y"] = 9 - action["from_y"]
        action["to_x"] = 8 - action["to_x"]
        action["to_y"] = 9 - action["to_y"]
    return actions


def get_all_actions(state, turn):
    if turn == c.RED:
        state = reverse_state(state)
    actions = []
    for y, line in enumerate(state):
        for x, piece_num in enumerate(line):
            if piece_num == 0 or piece_num[0] != turn:
                continue

            piece = piece_factory.get_piece(int(piece_num[1]))
            actions += piece.get_actions(state, x, y)
    if turn == c.RED:
        return reverse_actions(actions)
    return actions


def get_actions(state, x, y, turn):
    if state[y][x] == 0 or state[y][x][0] != turn:
        return None
    piece = piece_factory.get_piece(int(state[y][x][1]))
    if turn == c.RED:
        state = reverse_state(state)
        x = 8 - x
        y = 9 - y
    actions = piece.get_actions(state, x, y)
    if turn == c.RED:
        return reverse_actions(actions)
    return actions
