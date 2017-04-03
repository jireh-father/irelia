from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Constant(object):
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

    CONV_PIECE_LIST = {
        SOLDIER: '0.15',
        SANG: '0.29',
        GUARDIAN: '0.43',
        HORSE: '0.58',
        CANNON: '0.72',
        CAR: '0.86',
        KING: '1.0',
    }
