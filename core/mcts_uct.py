from math import sqrt, log
import numpy as np


class MctsUct(object):
    def __init__(self, env, num_iteration=1500, max_simulation=100, c_puct=2):
        self.env = env
        self.num_iteration = num_iteration
        self.max_simulation = max_simulation
        self.root_node = None
        self.current_node = None
        self.c_puct = c_puct

    def search(self, state, turn):
        self.root_node = Node(self.env, state, turn)
        self.current_node = self.root_node
        for i in range(self.num_iteration):
            print("iteration %d" % i)
            selected = True
            while selected:
                selected = self.select()
            self.expand()
            current_node = self.current_node
            root_value = self.simulation()
            self.current_node = current_node
            self.current_node.child_nodes = []
            if root_value is not False:
                self.update(root_value)

        for i, child_node in enumerate(self.root_node.child_nodes):
            print("child %d : visit - %f, wins - %f, turn - %s" % (
                i, float(child_node.visits), float(child_node.wins), child_node.turn), child_node.action)

        return sorted(self.root_node.child_nodes, key=lambda c: c.visits)[-1].action

    def select(self):
        if not self.current_node.child_nodes or self.current_node.untried_actions:
            return False

        node = self.current_node.select(self.c_puct)
        self.current_node = node
        return True

    @staticmethod
    def get_opponent_turn(turn):
        return 'b' if turn == 'r' else 'r'

    def expand(self):
        if not self.current_node.untried_actions:
            return
        legal_actions = self.current_node.untried_actions
        action_idx = np.random.choice(len(legal_actions), 1)[0]
        action = legal_actions[action_idx]
        next_state, info = self.env.simulate(self.current_node.state, action)
        next_node = Node(self.env, next_state, MctsUct.get_opponent_turn(self.current_node.turn), self.current_node,
                         action)
        self.current_node.child_nodes.append(next_node)
        del self.current_node.untried_actions[action_idx]
        self.current_node = next_node

    def simulation(self):
        is_game_over = False
        i = 0
        while not is_game_over:
            legal_actions = self.env.get_all_actions(self.current_node.state)
            if not legal_actions:
                print("lose turn no actions", self.current_node.turn)
                return -1 if self.current_node.turn == self.root_node.turn else 1
            action = legal_actions[np.random.choice(len(legal_actions), 1)[0]]
            next_state, info = self.env.simulate(self.current_node.state, action)
            next_node = Node(self.env, next_state, MctsUct.get_opponent_turn(self.current_node.turn),
                             self.current_node, action)
            self.current_node.child_nodes.append(next_node)
            self.current_node = next_node
            is_game_over = info['is_game_over']
            i += 1
            if self.max_simulation <= i:
                print("draw!")
                return False
                # winner = self.env.get_winner_by_point(self.current_node.state)
                # if not winner:
                #     return False
                # return 1 if winner == self.root_node.turn else -1
        print("lose turn", self.current_node.turn)
        return -1 if self.current_node.turn == self.root_node.turn else 1

    def update(self, root_value):
        node = self.current_node
        opponent_value = -1 if root_value == 1 else 1
        print("root_value", root_value)
        print("opponent_value", opponent_value)
        i = 0
        while node:
            if self.root_node.turn == node.turn:
                print("my turn update", root_value)
                node.wins -= root_value
            else:
                print("your turn update", opponent_value)
                node.wins += opponent_value

            node.visits += 1.
            i += 1
            node = node.parent_node
        self.current_node = self.root_node


class Node:
    def __init__(self, env, state, turn, parent_node=None, action=None):
        self.action = action
        self.turn = turn
        self.parent_node = parent_node
        self.child_nodes = []
        self.wins = 0.
        self.visits = 0.
        self.state = state
        self.untried_actions = env.get_all_actions(state)

    def select(self, c_puct):
        s = sorted(self.child_nodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        # s = sorted(self.child_nodes, key=lambda c: c.wins / c.visits + sqrt(c_puct) * sqrt(self.visits / c.visits))[-1]
        return s
