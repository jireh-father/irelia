from game.game import Game
import traceback


env = Game.make("KoreanChess-v1")

env.reset()

for i in range(100):
    while True:
        from_x = raw_input("from X number : ")
        if from_x.isdigit():
            from_x = int(from_x)
            break
    while True:
        from_y = raw_input("from Y number : ")
        if from_y.isdigit():
            from_y = int(from_y)
            break
    while True:
        to_x = raw_input("to X number : ")
        if to_x.isdigit():
            to_x = int(to_x)
            break
    while True:
        to_y = raw_input("to Y number : ")
        if to_y.isdigit():
            to_y = int(to_y)
            break

    try:
        new_state, reward, done, _ = env.step({"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y})

        if done:
            print("The End")
            break

    except Exception as e:
        print(e.message)
        traceback.print_exc()
