from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from game.game import Game
import numpy as np
import time
import json
from env import korean_chess_util as kcu

Game.register('KoreanChess',
              {'position_type': 'random', 'state_list': None,
               'init_state': None, 'init_side': 'b'})
env = Game.make('KoreanChess')
ret = env.convert_state_map(
    'r6,1,r2,r3,1,r3,r2,1,r6,4,r7,8,r5,r5,r4,6,r4,r1,r1,1,r1,2,r1,14,b6,b1,2,b1,b4,b1,b1,3,b5,b4,1,b5,8,b7,4,b6,1,b2,b3,1,b3,b2,2')
print(ret)
# blue_state = env.reset()
