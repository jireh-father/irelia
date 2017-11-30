# coding=utf8
import math
import numpy as np


class Mcts(object):
    def __init__(self, state, env, model, max_simulation=500, winner_reward=1., loser_reward=-1., c_puct=0.01,
                 init_root_edges=False):
        self.env = env
        self.model = model
        self.max_simulation = max_simulation
        self.root_node = Node(state)
        self.selected_edges = []
        self.action_history = []
        self.current_node = self.root_node
        self.temperature = .0
        self.winner_reward = winner_reward
        self.loser_reward = loser_reward
        self.c_puct = c_puct
        if init_root_edges:
            self.expand_and_evaluate()

    def search(self, temperature=.0, action_idx_list=None):
        self.temperature = temperature
        if action_idx_list is not None:
            if not self.root_node.edges:
                self.expand_and_evaluate()
            for action_idx in action_idx_list:
                self.root_node = self.root_node.edges[action_idx].node
        if self.root_node.edges is not None:
            # visits = []
            # for edge in self.root_node.edges:
            #     visits.append(edge.visit_count)
            # visits = np.array(visits)
            # visits = visits / visits.sum() * -1.
            # visits -= visits.min()
            # noise_probs = np.random.dirichlet(visits, 1)[0]
            noise_probs = np.random.dirichlet([1] * len(self.root_node.edges), 1)[0]
            for i, edge in enumerate(self.root_node.edges):
                edge.add_noise(noise_probs[i])

        for i in range(self.max_simulation):
            print("mcts simulate %d " % i)
            self.simulate()

        action_probs = np.array(
            [edge.get_action_probs(self.root_node.edges, self.temperature) for edge in self.root_node.edges])
        print("MCTS root edges")
        for i, edge in enumerate(self.root_node.edges):
            print("%d edge score! N: %d, P: %f, total_value: %f, mean_value: %f -> %s" % (
                i, edge.visit_count, edge.action_prob, edge.total_action_value, edge.mean_action_value,
                str(edge.action)))
        if (action_probs == 0).all():
            action_probs = np.array([1. / len(action_probs)] * len(action_probs))
        else:
            action_probs = action_probs / action_probs.sum() * 1.
        print("action probs!")
        print(action_probs)

        return action_probs

    def simulate(self):
        is_leaf_node = False
        i = 0
        while not is_leaf_node:
            print("mcts select %d" % i)
            is_leaf_node = self.select()
            if is_leaf_node == 2:
                self.backup(-1)
                self.init_state()
                return
            i += 1

        state_value = self.expand_and_evaluate()
        self.backup(state_value)

    def choice_edge_idx(self, select_scores):
        if (select_scores == 0).all():
            edge_idx = np.random.choice(len(select_scores), 1)[0]
        else:
            arg_max_list = np.argwhere(select_scores == np.amax(select_scores)).flatten()
            if len(arg_max_list) > 1:
                edge_idx = np.random.choice(arg_max_list, 1)[0]
            else:
                edge_idx = select_scores.argmax()
        return edge_idx

    def get_action_idx(self, action_probs):
        if self.temperature == 0:
            arg_max_list = np.argwhere(action_probs == np.amax(action_probs)).flatten()
            print("MCTS Max score:%f" % arg_max_list[0])
            if len(arg_max_list) > 1:
                action_idx = np.random.choice(arg_max_list, 1)[0]
            else:
                action_idx = action_probs.argmax()
        else:
            action_idx = np.random.choice(len(action_probs), 1, p=action_probs)[0]
        print("choice action idx %d" % action_idx)
        return action_idx

    def select(self):
        if not self.current_node.edges:
            return True
        select_scores = np.array(
            [edge.get_select_score(self.current_node.edges, self.c_puct) for edge in self.current_node.edges])
        edge_idx = self.choice_edge_idx(select_scores)

        if self.env.check_repeat(self.current_node.edges[edge_idx].action, self.action_history):
            if len(select_scores) == 1:
                return 2
            else:
                select_scores = np.delete(select_scores, edge_idx, 0)
                edge_idx = self.choice_edge_idx(select_scores)

        self.selected_edges.append(self.current_node.edges[edge_idx])
        self.action_history.append(self.current_node.edges[edge_idx].action)
        self.current_node = self.current_node.edges[edge_idx].node
        # self.env.print_env(state=self.current_node.state)

        return False

    def expand_and_evaluate(self):
        print("Expand and Evaluate!")
        if self.env.is_over(self.current_node.state):
            print("MCTS Game Over")
            return self.loser_reward

        # todo: !!!<일단 요것만구현>반복수 제한도 걸기(여러 반복수하면 비겨서 계산하는 조건도 알아보기)
        # todo :pass액션 추가 ( 둘다 pass할경우 점수계산으로
        action_probs, state_value = self.model.inference(self.current_node.state)
        print("MCTS Value inference", state_value)
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
        if (legal_action_probs == 0).all():
            legal_action_probs = np.array([1. / len(legal_action_probs)] * len(legal_action_probs))
        else:
            legal_action_probs = legal_action_probs / legal_action_probs.sum() * 1.
        # todo: add noise!! check (DIR(0.03)???)
        if self.root_node is self.current_node:
            # add noise to prior probabilities
            # if (legal_action_probs == 0).all():
            #     noise_probs = legal_action_probs
            # else:
            # noise_probs = np.random.dirichlet(legal_action_probs, 1)[0]
            noise_probs = np.random.dirichlet([1] * len(legal_action_probs), 1)[0]

            legal_action_probs = ((1 - 0.25) * legal_action_probs + (noise_probs * 0.25))

        self.current_node.edges = [
            Edge(action_prob, self.env.simulate(self.current_node.state, legal_actions[i], False), legal_actions[i]) for
            i, action_prob in enumerate(legal_action_probs)]

        return state_value

    def backup(self, state_value):
        print("MCTS Backup")
        self.selected_edges.reverse()
        for i, edge in enumerate(self.selected_edges):
            if i % 2 == 0:
                edge.update(-state_value)
            else:
                edge.update(state_value)
        self.init_state()

    def init_state(self):
        self.current_node = self.root_node
        self.selected_edges = []
        self.action_history = []

    def print_tree(self):
        print("========== mcts tree trace ==========")
        self.print_row([self.root_node])
        print("=====================================")

    def print_row(self, nodes, row_idx=0):
        child_nodes = []
        for node in nodes:
            for edge in node.edges:
                child_nodes.append(edge.node)
        print("%d row: %d nodes" % (row_idx, len(nodes)))
        if row_idx > 996:
            print("more...")
            return
        if child_nodes:
            self.print_row(child_nodes, row_idx + 1)


class Node(object):
    def __init__(self, state):
        self.state = state
        self.edges = []


class Edge(object):
    def __init__(self, action_prob, state, action):
        # N
        self.visit_count = .0
        # W
        self.total_action_value = .0
        # Q
        self.mean_action_value = .0
        # P
        self.action_prob = action_prob
        self.action = action
        self.node = Node(state)

    def add_noise(self, noice_prob):
        self.action_prob = (0.50 * self.action_prob) + (0.50 * noice_prob)

    def update(self, state_value):
        self.visit_count += 1.
        self.total_action_value += state_value
        self.mean_action_value = self.total_action_value / self.visit_count

    def get_select_score(self, edges, c_puct):
        # todo : what is b?? other acitions visit count?? check!!
        total_other_edge_visit_count = .0
        for edge in edges:
            total_other_edge_visit_count += edge.visit_count
        U = c_puct * self.action_prob * (math.sqrt(total_other_edge_visit_count) / (1. + self.visit_count))
        return self.mean_action_value + U

    def get_action_probs(self, edges, temperature):
        # todo : what is b?? other acitions visit count?? check!!
        total_other_edge_visit_count = .0
        for edge in edges:
            if temperature == 0:
                total_other_edge_visit_count += edge.visit_count
            else:
                total_other_edge_visit_count += (pow(edge.visit_count, 1. / temperature))
        if temperature == 0:
            return self.visit_count / total_other_edge_visit_count
        else:
            return pow(self.visit_count, 1. / temperature) / total_other_edge_visit_count
