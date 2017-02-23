from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env.korean_chess_piece import cannon
from env.korean_chess_piece import car
from env.korean_chess_piece import guardian
from env.korean_chess_piece import horse
from env.korean_chess_piece import king
from env.korean_chess_piece import sang
from env.korean_chess_piece import soldier
from env import korean_chess_util


def get_actions(state_map, x, y):
    piece_num = int(state_map[y][x][1])
    if piece_num == korean_chess_util.KING:
        return king.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.SOLDIER:
        return soldier.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.SANG:
        return sang.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.GUARDIAN:
        return guardian.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.HORSE:
        return horse.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.CANNON:
        return cannon.get_actions(state_map, x, y)
    elif piece_num == korean_chess_util.CAR:
        return car.get_actions(state_map, x, y)
