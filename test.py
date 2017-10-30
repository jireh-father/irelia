from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from game import korean_chess_constant
from game.korean_chess_v1 import KoreanChessV1
print(KoreanChessV1.default_state)
s = korean_chess_util.decode_state(KoreanChessV1.default_state)
print(s)
