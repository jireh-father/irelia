from math import sqrt, log


class MctsUct(object):
    def __init__(self, state, env, max_simulation=500, winner_reward=1., loser_reward=-1., c_puct=2):
        self.env = env
        self.max_simulation = max_simulation
        self.root_node = Node(state)
        self.state_history = [state]
        self.current_node = self.root_node
        self.winner_reward = winner_reward
        self.loser_reward = loser_reward
        self.c_puct = c_puct

    def search(self):
        for i in range(self.max_simulation):
            self.select()
            self.expand()
            value = self.simulation()
            self.update(value)


    def select(self):
        node = node.UCTSelectChild()
        pass

    def expand(self):
        pass

    def simulation(self):
        pass

    def update(self, value):
        pass

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parent_node = parent  # "None" for the root node
        self.child_nodes = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.GetMoves()  # future child nodes
        self.player_just_moved = state.playerJustMoved  # the only part of the state that the Node needs later

    def select(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.child_nodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def add_child(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untried_moves.remove(m)
        self.child_nodes.append(n)
        return n

    def update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(
            self.untried_moves) + "]"

    def tree_to_string(self, indent):
        s = self.indent_string(indent) + str(self)
        for c in self.child_nodes:
            s += c.TreeToString(indent + 1)
        return s

    def indent_string(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def children_to_string(self):
        s = ""
        for c in self.child_nodes:
            s += str(c) + "\n"
        return s
