# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from game.korean_chess_piece import cannon
from game.korean_chess_piece import car
from game.korean_chess_piece import guardian
from game.korean_chess_piece import horse
from game.korean_chess_piece import king
from game.korean_chess_piece import sang
from game.korean_chess_piece import soldier
from game import korean_chess_constant as c

PIECE_MAP = {c.KING: king, c.CANNON: cannon, c.CAR: car, c.GUARDIAN: guardian, c.HORSE: horse, c.SANG: sang,
             c.SOLDIER: soldier}


def get_piece(piece_num):
    return PIECE_MAP[piece_num]
