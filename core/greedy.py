class Greedy(object):
    def __init__(self, env):
        self.env = env
        self.root_node = None
        self.record = open("greedy_history", mode="w+")
        pass

    def search(self, state, action_idx=None):

        if self.root_node is None:
            i = 0
            self.root_node = Node(state, self.env.current_turn)
            stack = [self.root_node]
            while stack:
                current_node = stack.pop()
                if not current_node.extended:
                    print(i)
                    self.record.write(str(i)+"\n")
                    i += 1
                    legal_actions = self.env.get_all_actions(current_node.state)
                    for legal_action in legal_actions:
                        next_state, info = self.env.simulate(state, legal_action, True)

                        if info["is_game_over"]:
                            current_node.update(current_node.turn)
                        else:
                            current_node.child_nodes.append(
                                Node(next_state, 'r' if self.env.current_turn == 'b' else 'b', current_node))

                for child_node in current_node.child_nodes:
                    stack.append(child_node)
            self.record.close()
        if action_idx:
            self.root_node = self.root_node.child_nodes[action_idx]

        max_win = 0
        max_win_idx = -1
        for i, child_node in enumerate(self.root_node.child_nodes):
            if child_node.num_wins > max_win:
                max_win = child_node.num_wins
                max_win_idx = i
        self.root_node = self.root_node.child_nodes[max_win_idx]
        legal_actions = self.env.get_all_actions(state)
        return legal_actions[max_win_idx]


class Node(object):
    def __init__(self, state, turn, parent_node=None):
        self.state = state
        self.child_nodes = []
        self.extended = False
        self.num_wins = 0
        self.num_loses = 0
        self.parent_node = parent_node
        self.turn = turn
        pass

    def update(self, turn):
        if turn == self.turn:
            self.num_wins += 1
        else:
            self.num_loses += 1

        if self.parent_node is None:
            return
        self.parent_node.update(turn)
