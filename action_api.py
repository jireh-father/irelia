from env.korean_chess.core import Core
from flask import Flask, render_template
from flask import request
import json
import requests

from util import sl_policy_network

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
    state_key = Core.convert_state_key(state_map)
    response = requests.get('ec2-52-79-229-138.ap-northeast-2.compute.amazonaws.com',
                            {'state_key': state_key, 'color': side},
                            timeout=12)
    if response.status_code != 200:
        return json.dumps(
            {'error': True, 'msg': 'another server request error', 'response': json.loads(response.json())})
    return response.json()


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


@app.route("/action_by_slpn")
def action_by_slpn():
    state_key = request.args.get('state_key')
    color = request.args.get('color')

    retry_cnt = 3
    for i in range(retry_cnt):
        try:
            action = sl_policy_network.sampling_action(state_key, color,
                                                       '/home/model/korean_chess/sl_policy_network.ckpt')
            break
        except Exception as e:
            1
    if not action:
        json.dumps({'error': True})
    else:
        action_list = action.split('_')

        json.dumps({'x': action_list[0],
                    'y': action_list[1],
                    'to_x': action_list[2],
                    'to_y': action_list[3]})


if __name__ == "__main__":
    app.run()
