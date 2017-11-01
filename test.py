from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from game import korean_chess_util
from game.korean_chess_v1 import KoreanChessV1
print(KoreanChessV1.default_state)
state = [[[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9],
 [[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9]]
state[0][0][0] = 4
state[1][0][1] = 5
s = korean_chess_util.encode_state(state)
print(s)
