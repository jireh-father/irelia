from env.korean_chess.core import Core
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
        reverse_state_map = Core.reverse_state_map(state_map)
        db_name = './q_blue.db'
    else:
        reverse_state_map = state_map
        db_name = './q_red.db'

    state_key = Core.convert_state_key(reverse_state_map)

    conn = sqlite3.connect(db_name)

    c = conn.cursor()

    c.execute("SELECT quality_json FROM t_quality WHERE state_key='" + state_key + "'")

    result = c.fetchone()
    reversed_state_map = Core.reverse_state_map(state_map)
    actions = Core.get_actions(reversed_state_map, side)
    if result:
        if result[0] == '0':
            action = Core.similar_action(actions, state_key, side, c)
        else:
            q_values = json.loads(result[0])
            max_action = int(max(q_values.iteritems(), key=operator.itemgetter(1))[0])
            if len(actions) <= max_action:
                action = Core.similar_action(actions, state_key, side, c)
            else:
                action = actions[max_action]
    else:
        action = Core.similar_action(actions, state_key, side, c)

    c.close()
    return json.dumps(action)


@app.route("/actions")
def actions():
    state_map = request.args.get('state_map')
    side = request.args.get('side')
    if not state_map or side not in ('b', 'r'):
        return json.dumps({'error': True, 'msg': 'invalid params', 'data': {'state_map': state_map, 'side': side}})
    state_map = json.loads(state_map)
    filter_state_map(state_map)
    result = Core.get_actions(state_map, side)

    return json.dumps(result)


def error(msg, data=None):
    return json.dumps({'error': True, 'msg': msg, 'data': data})


if __name__ == "__main__":
    app.run()
