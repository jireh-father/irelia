# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from game.korean_chess_piece import piece_factory
from game import korean_chess_constant as c
import numpy as np


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


def encode_state(state):
    new_state = [[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9]
    for i, line in enumerate(state[0]):
        for j, piece in enumerate(line):
            if piece != 0:
                new_state[i][j] = "b" + str(int(piece))
            if state[1][i][j] != 0:
                new_state[i][j] = "r" + str(int(state[1][i][j]))
    turn = c.BLUE if state[2][0][0] == 1 else c.RED
    return new_state, turn


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
        new_state.append([[-1] * 9, [-1] * 9, [-1] * 9, [-1] * 9, [-1] * 9, [-1] * 9, [-1] * 9, [-1] * 9, [-1] * 9, [-1] * 9])
    new_state = np.array(new_state).astype(np.float)
    return new_state


def validate_action(action, state, turn, next_turn, use_check=True):
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

    if not use_check:
        return True

    # check this action gets my own check.
    # todo: modify for oppnent turn
    check = is_check(state, from_x, from_y, to_x, to_y, next_turn)
    if check:
        raise Exception("this action causes opponent's check %d %d %d %d" % (from_x, from_y, to_x, to_y,))
        # return False

    return True


def check_repeat(action, action_history):
    my_actions = []
    for i, tmp_action in enumerate(action_history):
        if i % 2 == 1:
            continue
        my_actions.append(tmp_action)
    my_actions.reverse()
    even_action = my_actions[0]
    for i in range(1, len(my_actions)):
        if i % 2 == 0 and even_action == my_actions[i]:
            continue
        elif i % 2 == 1 and action == my_actions[i]:
            continue
        return False
    return True


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
        # for line in state:
        #     print(line)
        # get my actions after opponent's moving
        next_my_actions = get_all_actions(state, turn)
        # count my check
        check_cnt = 0
        for action in next_my_actions:
            # print("try", action)
            if state[action["to_y"]][action["to_x"]] != 0 and int(state[action["to_y"]][action["to_x"]][1]) == c.KING:
                # if is_check(state, action["from_x"], action["from_y"], action["to_x"], action["to_y"], turn):
                # print("check")
                check_cnt += 1
                # else:
                #     print("not check")
        if check_cnt == 0:
            # print("not checkmate")
            return False
        # get back to previous state
        state[from_y][from_x] = state[to_y][to_x]
        state[to_y][to_x] = old
    return True


def is_draw(state):
    other_piece_list = []
    for y, line in enumerate(state):
        for x, piece in enumerate(line):
            if piece == 0:
                continue
            if int(piece[1]) != c.KING and int(piece[1]) != c.GUARDIAN:
                other_piece_list.append([x, y])

    cannon_cnt = 0
    disable_cannon_cnt = 0
    for other_piece in other_piece_list:
        if int(state[other_piece[1]][other_piece[0]][1]) != c.CANNON:
            return False
        else:
            cannon_cnt += 1
            actions = get_actions(state, other_piece[0], other_piece[1], state[other_piece[1]][other_piece[0]][0])
            if not actions:
                disable_cannon_cnt += 1

    if cannon_cnt == disable_cannon_cnt:
        return True
    else:
        return False


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
