from math import sqrt, log
import numpy as np
import copy


class MctsUct(object):
    def __init__(self, env, num_iteration=1500, max_simulation=100, c_puct=2):
        self.env = env
        self.num_iteration = num_iteration
        self.max_simulation = max_simulation
        self.root_node = None
        self.current_node = None
        self.c_puct = c_puct

    def search(self, state, turn):
        self.root_node = Node(state, turn)
        self.current_node = self.root_node
        for i in range(self.num_iteration):
            print("iteration %d" % i)
            selected = True
            while selected:
                selected = self.select()
            self.expand()
            current_node = self.current_node
            value = self.simulation()
            self.current_node = current_node
            self.current_node.child_nodes = []
            if value >= 0:
                self.update(value)
        return sorted(self.root_node.child_nodes, key=lambda c: c.visits)[-1].action

    def select(self):
        if not self.current_node.child_nodes:
            return False

        node = self.current_node.select(self.c_puct)
        self.current_node = node
        return True

    @staticmethod
    def get_opponent_turn(turn):
        return 'b' if turn == 'r' else 'r'

    def expand(self):
        legal_actions = self.env.get_all_actions(self.current_node.state)
        if not legal_actions:
            return
        action = legal_actions[np.random.choice(len(legal_actions), 1)[0]]
        next_state, info = self.env.simulate(self.current_node.state, action)
        next_node = Node(next_state, MctsUct.get_opponent_turn(self.current_node.turn), self.current_node, action)
        self.current_node.child_nodes.append(next_node)
        self.current_node = next_node

    def simulation(self):
        is_game_over = False
        i = 0
        while not is_game_over:
            legal_actions = self.env.get_all_actions(self.current_node.state)
            if not legal_actions:
                return 0 if self.root_node.turn == self.current_node.turn else 1
            action = legal_actions[np.random.choice(len(legal_actions), 1)[0]]
            next_state, info = self.env.simulate(self.current_node.state, action)
            next_node = Node(next_state, MctsUct.get_opponent_turn(self.current_node.turn),
                             self.current_node, action)
            self.current_node.child_nodes.append(next_node)
            self.current_node = next_node
            is_game_over = info['is_game_over']
            i += 1
            if self.max_simulation <= i:
                return -1
        return 0 if self.root_node.turn == self.current_node.turn else 1

    def update(self, value):
        node = self.current_node
        i = 0
        while node:
            if node.turn == self.root_node.turn:
                node.wins += value
            else:
                node.wins += -value
            node.visits += 1.
            i += 1
            node = node.parent_node
        self.current_node = self.root_node


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, state, turn, parent_node=None, action=None):
        self.action = action
        self.turn = turn
        self.parent_node = parent_node
        self.child_nodes = []
        self.wins = 0.
        self.visits = 0.
        self.state = state

    def select(self, c_puct):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.child_nodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        # s = sorted(self.child_nodes, key=lambda c: c.wins / c.visits + sqrt(c_puct) * sqrt(self.visits / c.visits))[-1]
        return s