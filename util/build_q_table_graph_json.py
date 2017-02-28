import json
import numpy as np
import sys

q_state_link = open('../q_state_link.txt')
state_links = {}
for line in q_state_link:
    link = json.loads(line.strip())
    for key in link:
        for action in link[key]:
            if key not in state_links:
                state_links[key] = {}
            state_links[key][action] = link[key][action]

Q_blue = {}
Q_red = {}
q_file = open('../q_blue_with_data.txt')
i = 0
key = None
for line in q_file:
    if i % 2 is 0:
        key = line.strip()
    else:
        Q_blue[key] = np.array(json.loads(line.strip()))
    i += 1

q_file = open('../q_red_with_data.txt')
i = 0
key = None
for line in q_file:
    if i % 2 is 0:
        key = line.strip()
    else:
        Q_red[key] = np.array(json.loads(line.strip()))
    i += 1


def convert_state_key(state_map):
    return str(state_map).replace('[', '').replace(', ', ',').replace(']', '').replace("'", '')


def reverse_state_key(state):
    return convert_state_key(list(reversed(state.split(','))))


print(len(Q_blue), len(Q_red))
sys.exit()
graph_nodes = []
graph_links = []
state_list = {}
j = 0
# print(json.dumps(state_links))
for state_key in state_links:
    for action in state_links[state_key]:
        if state_key in Q_blue:
            target_state_key = state_links[state_key][action]
            value = int(Q_blue[state_key][int(action)] + 1)
            if value < 1:
                value = 1
            if state_key not in state_list:
                state_list[state_key] = True
                graph_nodes.append({'id': state_key, 'group': 1})
            if target_state_key not in state_list:
                state_list[target_state_key] = True
                graph_nodes.append({'id': target_state_key, 'group': 1})
            graph_links.append({'source': state_key, 'target': target_state_key, 'value': value})

for state_key in state_links:
    for action in state_links[state_key]:
        if state_key in Q_red:
            target_state_key = state_links[state_key][action]
            value = int(Q_red[state_key][int(action)] + 1)
            if value < 1:
                value = 1
            if state_key not in state_list:
                state_list[state_key] = True
                graph_nodes.append({'id': state_key, 'group': 1})
            if target_state_key not in state_list:
                state_list[target_state_key] = True
                graph_nodes.append({'id': target_state_key, 'group': 1})
            graph_links.append({'source': state_key, 'target': target_state_key, 'value': value})

with open('./q_table_graph.json', 'w') as outfile:
    outfile.write(json.dumps({'nodes': graph_nodes, 'links': graph_links}))
    outfile.close()
