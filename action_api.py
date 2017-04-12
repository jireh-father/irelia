from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env.korean_chess.core import Core
from flask import Flask, render_template
from flask import request
import json
import requests

import imp

try:
    from util import sl_policy_network
    import tensorflow as tf
    from util import neural_network as nn

    data_format = 'NHWC'
    width = 9
    height = 10
    num_input_feature = 3

    inputs = tf.placeholder(tf.float16, [None, height, width, num_input_feature], name='inputs')

    logits, end_points = nn.sl_policy_network(inputs, 11, 192, data_format=data_format)

    argmax = tf.argmax(end_points['Predictions'], 2)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()
    saver.restore(sess, '/home/model/korean_chess/sl_policy_network.ckpt')
except ImportError:
    1

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
    if side == 'b':
        state_key = Core.reverse_state_key(state_key)
    response = requests.get('http://52.79.229.138:5000/action_by_slpn',
                            {'state_key': state_key, 'color': side},
                            timeout=12)
    if response.status_code != 200:
        return json.dumps(
            {'error': True, 'msg': 'another server request error', 'response': json.loads(response.text)})
    return response.text


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
    sample_action = None
    for i in range(retry_cnt):
        try:
            sample_action = sl_policy_network.sampling_action(state_key, color, 'NHWC', True, sess, argmax, end_points,
                                                              inputs)
            break
        except Exception as e:
            print('exception: ' + str(e))
    if not sample_action:
        print('nothing action')
        return json.dumps({'error': True})
    else:
        action_list = sample_action.split('_')
        if color == 'b':
            result = json.dumps({'x': 8 - int(action_list[0]),
                                 'y': 9 - int(action_list[1]),
                                 'to_x': 8 - int(action_list[2]),
                                 'to_y': 9 - int(action_list[3])})
        else:
            result = json.dumps({'x': action_list[0],
                                 'y': action_list[1],
                                 'to_x': action_list[2],
                                 'to_y': action_list[3]})
        print(result)
        return result


if __name__ == "__main__":
    app.run()
