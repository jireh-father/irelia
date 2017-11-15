import math
import numpy as np


class Mcts(object):
    def __init__(self, state, env, model, max_simulation=500, winner_reward=1, loser_reward=-1, use_best=True,
                 c_puct=0.5):
        self.env = env
        self.model = model
        self.max_simulation = max_simulation
        self.root_node = Node(state)
        self.selected_edges = []
        self.current_node = self.root_node
        self.root_turn = None
        self.current_turn = None
        self.temperature = 0
        self.winner_reward = winner_reward
        self.loser_reward = loser_reward
        self.use_best = use_best
        self.c_puct = c_puct

    def search(self, temperature=0, action_idx=None):
        self.temperature = temperature
        self.root_turn = self.env.current_turn
        self.current_turn = self.env.current_turn
        if action_idx:
            if not self.root_node.edges:
                return False
            self.root_node = self.root_node.edges[action_idx].node

        for i in range(self.max_simulation):
            self.simulate()

        action_probs = [edge.get_action_probs(self.root_node.edges, self.temperature) for edge in self.root_node.edges]

        if self.use_best:
            self.root_node = self.root_node.edges[np.array(action_probs).argmax()].node
        return np.array(action_probs)

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
            score = edge.get_select_score(self.current_node.edges, self.c_puct)
            if max_score < score:
                max_score = score
                max_score_edge = i

        self.current_node = self.current_node.edges[max_score_edge].node
        self.selected_edges.append(self.current_node.edges[max_score_edge])
        self.current_turn = 'r' if self.current_turn == 'b' else 'b'

        return False

    def expand_and_evaluate(self):
        if self.env.is_over(self.current_node.state):
            return self.winner_reward

        # todo: implement model class
        # todo: !!!<일단 요것만구현>반복수 제한도 걸기(여러 반복수하면 비겨서 계산하는 조건도 알아보기)
        # todo :pass액션 추가 ( 둘다 pass할경우 점수계산으로
        action_probs, state_value = self.model.inference(self.current_node.state)

        # todo : <<빅장>> 혹은 외통수(장군)등 기능 구현?
        # todo: 비긴 상태 구현해서 적용하기(더 디테일하게)

        legal_actions = self.env.get_all_actions(self.current_node.state)

        if not legal_actions:
            return self.loser_reward

        legal_action_probs = []
        for legal_action in legal_actions:
            legal_action = self.env.encode_action(legal_action)
            legal_action_probs.append(action_probs[legal_action[0]] + action_probs[legal_action[0]])

        legal_action_probs = np.array(legal_action_probs)
        # todo: add noise!! check (DIR(0.03)???)
        if self.root_node is self.current_node:
            # add noise to prior probabilities
            noise_probs = np.random.dirichlet(legal_action_probs, 1)[0]
            legal_action_probs = ((1 - 0.25) * legal_action_probs + (noise_probs * 0.25))

        self.current_node.edges = [
            Edge(action_prob, self.env.simulate(self.current_node.state, legal_actions[i]), legal_actions[i]) for
            i, action_prob in enumerate(legal_action_probs)]

        return -state_value if self.current_turn != self.root_turn else state_value

    def backup(self, state_value):
        self.selected_edges.reverse()
        for i, edge in enumerate(self.selected_edges):
            if i % 2 == 0:
                state_value = -state_value
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
