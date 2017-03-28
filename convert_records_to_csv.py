import json
import csv
from game.game import Game

records_path = 'D:/data/korean_chess/records.txt'
output_path = 'D:/data/korean_chess/records.csv'
k = 0
Game.register('KoreanChess',
              {'position_type': 'random', 'init_state': None, 'init_side': 'b'})
env = Game.make('KoreanChess')

output_file = open(output_path, mode='w')

records_file = open(records_path)
lines = records_file.readlines()
for i, line in enumerate(lines):
    if i > 100:
        break
    line = line.strip()
    records = json.loads(line)
    record_count = len(records['records'])
    env.set_property('position_type', [records['blue_position_type'], records['red_position_type']])
    records = records['records']

    state = env.reset()

    j = 0
    while j <= record_count:
        print(i, j)
        conv_input = env.build_csv_data(state, 'b', records[j])
        output_file.write(','.join(conv_input) + "\n")

        action = env.get_action_with_record({}, state, records[j])
        state, _, _, _ = env.step(action, state)
        # red action
        j += 1
        if j >= record_count:
            break

        conv_input = env.build_csv_data(state, 'r', records[j])
        output_file.write(','.join(conv_input) + "\n")

        action = env.get_action_with_record({}, state, records[j], True)
        state, _, _, _ = env.step(action, state, True)

        j += 1
        if j >= record_count:
            break

output_file.close()
