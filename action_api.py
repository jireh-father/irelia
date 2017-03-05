from env.korean_chess import KoreanChess
from flask import Flask
from flask import request
import json
import sqlite3

app = Flask(__name__, static_url_path='/home/irelia/public_html')


@app.route('/play')
def play():
    return app.send_static_file('web/play.html')


@app.route("/action")
def action():
    state_map = request.args.get('state_map')

    side = request.args.get('side')
    if not state_map or side not in ('b', 'r'):
        return json.dumps({'error': True, 'msg': 'invalid params', 'data': {'state_map': state_map, 'side': side}})

    state_map = json.loads(state_map)

    if side is 'b':
        state_map = KoreanChess.reverse_state_map(state_map)
        db_name = 'q_blue.db'
    else:
        db_name = 'q_red.db'

    state_key = KoreanChess.convert_state_key(state_map)

    conn = sqlite3.connect(db_name)

    c = conn.cursor()

    c.execute("SELECT * FROM t_quality WHERE state_key='" + state_key + "'")

    result = c.fetchone()
    c.close()
    return result

    # return "Hello World!"


@app.route("/actions")
def actions():
    state_map = request.args.get('state_map')
    side = request.args.get('side')
    if not state_map or side not in ('b', 'r'):
        return json.dumps({'error': True, 'msg': 'invalid params', 'data': {'state_map': state_map, 'side': side}})
    result = KoreanChess.get_actions(json.loads(state_map), side)

    return json.dumps(result)


if __name__ == "__main__":
    app.run()
