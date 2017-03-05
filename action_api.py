from env.korean_chess import KoreanChess
from flask import Flask
from flask import request
import json

app = Flask(__name__, static_url_path='')

@app.route('/play')
def play():
    return app.send_static_file('web/play.html')

@app.route("/action")
def action():
    return "Hello World!"


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
