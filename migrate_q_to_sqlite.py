import sqlite3
import os
import json
import numpy as np
import datetime
from env.korean_chess.core import Core

conn = sqlite3.connect('./q_blue.db')

c = conn.cursor()

# Create table
c.execute(
    "CREATE TABLE IF NOT EXISTS t_quality ( state_key text PRIMARY KEY, quality_json text, action_size integer NOT NULL, update_date text, update_cnt integer DEFAULT 1 );")

q_file = open('./q_blue_with_data.txt')
i = 0
key = None
for line in q_file:
    if i % 2 is 0:
        key = line.strip()
    else:
        q_value = np.array(json.loads(line.strip()))
        if np.sum(q_value) == 0:
            quality_json = '0'
        else:
            quality_dict = {}
            for j, q in enumerate(q_value):
                if q != 0:
                    quality_dict[j] = q
            quality_json = json.dumps(quality_dict)

        # Insert a row of data
        c.execute("INSERT INTO t_quality VALUES " + "('%s', '%s', %d, '%s', %d)" % ((Core.compress_state_key(key), quality_json, len(q_value),
                                             datetime.datetime.now().strftime('%Y-%m-%d %H:%I:%S'), 1)))
    i += 1

q_file.close()


# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
conn = sqlite3.connect('./q_red.db')

c = conn.cursor()

# Create table
c.execute(
    "CREATE TABLE IF NOT EXISTS t_quality ( state_key text PRIMARY KEY, quality_json text, action_size integer NOT NULL, update_date text, update_cnt integer DEFAULT 1 );")

q_file = open('./q_red_with_data.txt')
i = 0
key = None
for line in q_file:
    if i % 2 is 0:
        key = line.strip()
    else:
        q_value = np.array(json.loads(line.strip()))
        if np.sum(q_value) == 0:
            quality_json = '0'
        else:
            quality_dict = {}
            for j, q in enumerate(q_value):
                if q != 0:
                    quality_dict[j] = q
            quality_json = json.dumps(quality_dict)

        c.execute("INSERT INTO t_quality VALUES " + "('%s', '%s', %d, '%s', %d)" % ((Core.compress_state_key(key), quality_json, len(q_value),
                                         datetime.datetime.now().strftime('%Y-%m-%d %H:%I:%S'), 1)))
    i += 1


q_file.close()


# Insert a row of data


# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()