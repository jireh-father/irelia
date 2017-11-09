from game.game import Game
import traceback
from game import korean_chess_constant as c
import random
from builtins import input

init_state = [[[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9],
              [[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9],
              [[1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9, [1] * 9]]
init_state[1][1][3] = c.KING
init_state[0][8][5] = c.KING
init_state[0][6][4] = c.CAR
init_state[0][5][4] = c.CAR
# env = Game.make("KoreanChess-v1", {"init_state": init_state})
# env = Game.make("KoreanChess-v1", {"interval": 0.1})
env = Game.make("KoreanChess-v1", {"use_check": False, "limit_step": 400, "interval": 2})
episodes = 1000
for i in range(episodes):
    env.reset()
    while True:
        actions = env.get_all_actions()
        action = actions[random.randint(0, len(actions) - 1)]
        try:
            new_state, reward, done, _ = env.step(
                {"from_x": action["from_x"], "from_y": action["from_y"], "to_x": action["to_x"],
                 "to_y": action["to_y"]})

            if done:
                print("The End")
                break
        except Exception as e:
            # print(e)
            env.print_env()
            traceback.print_exc()
            continue
