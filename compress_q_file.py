import sqlite3
import os
import json
import numpy as np
import datetime
from env.korean_chess import KoreanChess


q_file = open('./q_blue_with_data.txt')
compress_q_file = open('./q_blue_with_data_comp.txt', mode='w')
i = 0
key = None
for line in q_file:
    if i % 2 is 0:
        key = line.strip()
        compress_q_file.write(KoreanChess.compress_state_key(key) + "\n")
    else:
        compress_q_file.write(line.strip() + "\n")
    i += 1

compress_q_file.close()
q_file.close()



q_file = open('./q_red_with_data.txt')
compress_q_file = open('./q_red_with_data_comp.txt', mode='w')
i = 0
key = None
for line in q_file:
    if i % 2 is 0:
        key = line.strip()
        compress_q_file.write(KoreanChess.compress_state_key(key) + "\n")
    else:
        compress_q_file.write(line.strip() + "\n")
    i += 1
    i += 1


compress_q_file.close()
q_file.close()

