from env.korean_chess import KoreanChess
from flask import Flask, render_template
from flask import request
import json
import sqlite3
import operator
import random
import numpy as np
from elasticsearch import Elasticsearch as ES

app = Flask(__name__)


@app.route('/<string:page_name>/')
def static_page(page_name):
    return render_template('%s.html' % page_name)


def filter_state_map(state_map):
    for y, row in enumerate(state_map):
        for x, piece in enumerate(row):
            if piece != 0:
                state_map[y][x] = str(piece)


@app.route("/action")
def action():
    state_map = request.args.get('state_map')

    side = request.args.get('side')
    if not state_map or side not in ('b', 'r'):
        return json.dumps({'error': True, 'msg': 'invalid params', 'data': {'state_map': state_map, 'side': side}})

    state_map = json.loads(state_map)
    filter_state_map(state_map)
    if side == 'b':
        reverse_state_map = KoreanChess.reverse_state_map(state_map)
        db_name = './q_blue.db'
    else:
        reverse_state_map = state_map
        db_name = './q_red.db'

    state_key = KoreanChess.convert_state_key(reverse_state_map)

    conn = sqlite3.connect(db_name)

    c = conn.cursor()

    c.execute("SELECT quality_json FROM t_quality WHERE state_key='" + state_key + "'")

    result = c.fetchone()
    reversed_state_map = KoreanChess.reverse_state_map(state_map)
    actions = KoreanChess.get_actions(reversed_state_map, side)
    if result:
        if result[0] == '0':
            action = similar_action(actions, state_key, side, c)
        else:
            q_values = json.loads(result[0])
            max_action = int(max(q_values.iteritems(), key=operator.itemgetter(1))[0])
            if len(actions) <= max_action:
                action = similar_action(actions, state_key, side, c)
            else:
                action = actions[max_action]
    else:
        action = similar_action(actions, state_key, side, c)

    c.close()
    return json.dumps(action)


def similar_action(actions, state_key, side, sqlite_cursor):
    decomp_state_key = KoreanChess.decompress_state_key(state_key)
    es = ES('52.79.166.102:80')
    result = es.search('i_irelia_state', 't_blue_state' if side is 'b' else 't_red_state',
                       {
                           "query": {"match": {
                               "state": decomp_state_key}}
                       })
    if not result or result['_shards']['failed'] > 0:
        return random.choice(actions)

    for item in result['hits']['hits']:
        q_state = item['_source']['state']
        sqlite_cursor.execute(
            "SELECT quality_json FROM t_quality WHERE state_key='" + KoreanChess.compress_state_key(q_state) + "'")

        q_result = sqlite_cursor.fetchone()
        if not q_result or q_result[0] == '0':
            continue

        q_values = json.loads(result[0])
        max_action = int(max(q_values.iteritems(), key=operator.itemgetter(1))[0])
        state_map = KoreanChess.convert_state_map(state_key)
        if side == 'r':
            state_map = KoreanChess.reverse_state_map(state_map)
        new_actions = KoreanChess.get_actions(state_map, side)
        if len(new_actions) <= max_action:
            action = random.choice(actions)
        else:
            action = new_actions[max_action]

        return action
    return random.choice(actions)


@app.route("/actions")
def actions():
    state_map = request.args.get('state_map')
    side = request.args.get('side')
    if not state_map or side not in ('b', 'r'):
        return json.dumps({'error': True, 'msg': 'invalid params', 'data': {'state_map': state_map, 'side': side}})
    state_map = json.loads(state_map)
    filter_state_map(state_map)
    result = KoreanChess.get_actions(state_map, side)

    return json.dumps(result)


def error(msg, data=None):
    return json.dumps({'error': True, 'msg': msg, 'data': data})


if __name__ == "__main__":
    app.run()
