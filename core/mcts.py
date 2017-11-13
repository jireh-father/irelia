import math
import numpy as np


class Mcts(object):
    def __init__(self, state, env, model, maximum_simulate, is_train=True):
        self.env = env
        self.model = model
        self.maximum_simulate = maximum_simulate
        self.root_node = Node(state)
        self.selected_edges = []
        self.current_node = self.root_node
        self.current_turn = None
        self.temperature = 0
        self.is_train = is_train

    def search(self, current_turn, temperature=0):
        self.temperature = temperature
        self.current_turn = current_turn
        for i in range(self.maximum_simulate):
            self.simulate()
        if not self.root_node.edges:
            return False

        action_probs = [edge.get_action_probs(self.root_node.edges, self.temperature) for edge in self.root_node.edges]

        self.root_node = self.root_node.edges[np.array(action_probs).argmax()].node
        return self.root_node.edges[np.array(action_probs).argmax()].action

    def simulate(self):
        is_leaf_node = False
        while not is_leaf_node:
            is_leaf_node = self.select()

        state_value = self.expand_and_evaluate()
        self.backup(state_value)

    def select(self):
        if not self.current_node.edges:
            return True
        max_score = 0
        max_score_edge = -1
        for i, edge in enumerate(self.current_node.edges):
            score = edge.get_select_score(self.current_node.edges, 1)
            if max_score < score:
                max_score = score
                max_score_edge = i

        self.current_node = self.current_node.edges[max_score_edge].node
        self.selected_edges.append(self.current_node.edges[max_score_edge])

        return False

    def expand_and_evaluate(self):
        # todo: implement model class
        action_probs, state_value = self.model.inference(self.current_node.state)
        # todo: add noise!!
        if self.root_node is self.current_node:
            # add noise to prior probabilities
            action_probs = ((1 - 0.25) * action_probs + 0.25 * 0.03)

        if self.env.is_over(self.current_node.state):
            # todo: implement env.winner function
            return 1 if self.env.get_winner() == self.current_turn else 0

        legal_actions = self.env.get_all_actions(self.current_node.state, self.current_turn)

        legal_action_probs = []
        for legal_action in legal_actions:
            legal_action = self.env.encode_action(legal_action)
            legal_action_probs.append(action_probs[legal_action[0]] + action_probs[legal_action[0]])

        # todo: implement env.simulate
        self.current_node.edges = [
            Edge(action_prob, self.env.simulate(self.current_node.state, legal_actions[i]), legal_actions[i]) for
            i, action_prob in enumerate(legal_action_probs)]

        return state_value

    def backup(self, state_value):
        # todo: change to me and 상대 점수 다르게 세팅되도록
        for edge in self.selected_edges:
            edge.update(state_value)

        self.current_node = self.root_node
        self.selected_edges = []


class Node(object):
    def __init__(self, state):
        self.state = state
        self.edges = []


class Edge(object):
    def __init__(self, action_prob, state, action):
        # N
        self.visit_count = 0
        # W
        self.total_action_value = 0
        # Q
        self.mean_action_value = 0
        # P
        self.action_prob = action_prob
        self.action = action
        self.node = Node(state)

    def update(self, state_value):
        self.visit_count += 1
        self.total_action_value += state_value
        self.mean_action_value = self.total_action_value / self.visit_count

    def get_select_score(self, edges, c_puct):
        # todo : what is b?? other acitions visit count?? check!!
        total_other_edge_visit_count = 0
        for edge in edges:
            total_other_edge_visit_count += edge.visit_count
        U = c_puct * self.action_prob * (math.sqrt(total_other_edge_visit_count) / (1 + self.visit_count))
        return self.mean_action_value + U

    def get_action_probs(self, edges, temperature):
        # todo : what is b?? other acitions visit count?? check!!
        total_other_edge_visit_count = 0
        for edge in edges:
            total_other_edge_visit_count += (pow(edge.visit_count, 1 / temperature))
        return pow(self.visit_count, 1 / temperature) / total_other_edge_visit_count
