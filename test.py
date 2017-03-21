from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sqlite3
import sys

conn = sqlite3.connect('q_red.db')

c = conn.cursor()

c.execute("SELECT quality_json FROM t_quality WHERE state_key='r6,r4,r2,r3,1,r3,r4,r2,r6,4,r7,5,r5,5,r5,1,r1,1,r1,1,r1,1,r1,1,r1,18,b1,1,b1,1,b1,1,b1,b1,2,b5,5,b5,5,b7,4,b6,b4,b2,b3,1,b3,b4,b2,b6'")

result = c.fetchone()
print(result)
sys.exit()
ret = Core.convert_state_map(
    'b6,b2,b4,b3,1,b3,b2,b4,b6,4,b7,5,b5,5,b5,1,b1,1,b1,1,b1,1,b1,b1,19,r1,1,r1,1,r1,1,r1,1,r1,1,r5,5,r5,5,r7,4,r6,r4,r2,r3,1,r3,r2,r4,r6')
print(ret)
sys.exit()
Game.register('KoreanChess',
              {'position_type': 'random', 'state_list': None,
               'init_state': None, 'init_side': 'b'})
env = Game.make('KoreanChess')


ret = env.convert_state_map(
    'r6,1,r2,r3,1,r3,r2,1,r6,4,r7,8,r5,r5,r4,6,r4,r1,r1,1,r1,2,r1,14,b6,b1,2,b1,b4,b1,b1,3,b5,b4,1,b5,8,b7,4,b6,1,b2,b3,1,b3,b2,2')
print(ret)
# blue_state = env.reset()
