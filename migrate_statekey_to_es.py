from elasticsearch import Elasticsearch as ES
from env.korean_chess import KoreanChess
import sys

es = ES('52.79.135.2:80')

q_file = open('./q_blue_with_data.txt')
i = 0
key = None
bulk_list = []
blue_index = {"index": {"_index": "i_irelia_state", "_type": "t_blue_state"}}
for line in q_file:
    if i % 2 is 0:
        key = line.strip()
        decomp_key = KoreanChess.decompress_state_key(key)
        bulk_list.append(blue_index)
        bulk_list.append({"state": decomp_key})
    if i % 1000 == 0 and i != 0:
        es.bulk(bulk_list)
        bulk_list = []
    i += 1

q_file.close()
es.bulk(bulk_list)

q_file = open('./q_red_with_data.txt')
i = 0
key = None
bulk_list = []
red_index = {"index": {"_index": "i_irelia_state", "_type": "t_red_state"}}
for line in q_file:
    if i % 2 is 0:
        key = line.strip()
        decomp_key = KoreanChess.decompress_state_key(key)
        bulk_list.append(red_index)
        bulk_list.append({"state": decomp_key})
    if i % 1000 == 0 and i != 0:
        es.bulk(bulk_list)
        bulk_list = []
    i += 1

q_file.close()

es.bulk(bulk_list)
