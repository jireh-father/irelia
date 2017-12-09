# coding=utf8
import math
import numpy as np
from util import common
import time


class Mcts(object):
    START = []

    def __init__(self, state, env, model, max_simulation=500, winner_reward=1., loser_reward=-1., c_puct=0.01,
                 init_root_edges=False, num_state_history=7, print_mcts_search=True):
        self.env = env
        self.model = model
        self.max_simulation = max_simulation
        self.root_node = Node(state)
        self.selected_edges = []
        self.action_history = []
        self.state_history = [state]
        self.current_node = self.root_node
        self.temperature = .0
        self.winner_reward = winner_reward
        self.loser_reward = loser_reward
        self.c_puct = c_puct
        self.num_state_history = num_state_history
        self.print_mcts_search = print_mcts_search
        self.start = 0
        if init_root_edges:
            self.expand_and_evaluate()

    def log(self, *args):
        if self.print_mcts_search:
            print(args)

    @staticmethod
    def te(msg=None):
        return
        if msg:
            print("time: ", msg, time.time() - Mcts.START.pop())
        else:
            Mcts.START.append(time.time())

    def search(self, temperature=.0, action_idx_list=[]):
        self.temperature = temperature
        if len(action_idx_list) > 0:
            for action_idx in action_idx_list:
                if not self.root_node.edges:
                    self.expand_and_evaluate()
                self.root_node = self.root_node.edges[action_idx].node
            self.root_node.parent_edge = None
            self.root_node.parent_node = None
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
            self.log("mcts simulate %d " % i)

            self.simulate()

        action_probs = np.array(
            [edge.get_action_probs(self.root_node.edges, self.temperature) for edge in self.root_node.edges])
        self.log("MCTS root edges")

        for i, edge in enumerate(self.root_node.edges):
            self.log("%d edge score! N: %d, P: %f, total_value: %f, mean_value: %f -> %s" % (
                i, edge.visit_count, edge.action_prob, edge.total_action_value, edge.mean_action_value,
                str(edge.action)))

        if (action_probs == 0).all():

            action_probs = np.array([1. / len(action_probs)] * len(action_probs))

        else:

            action_probs = action_probs / action_probs.sum()

        self.log("action probs!")
        self.log(action_probs)

        return action_probs

    def simulate(self):
        is_leaf_node = False
        i = 0
        while not is_leaf_node:
            self.log("mcts select %d" % i)

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

    def choice_no_visited_edge_idx(self, skip_idx=None):
        no_visited_idx_list = []

        for i, edge in enumerate(self.current_node.edges):
            if edge.visit_count == 0:
                no_visited_idx_list.append(i)

        if len(no_visited_idx_list) == 0:
            return None
        if skip_idx is None:

            edge_idx = np.random.choice(no_visited_idx_list, 1)[0]

        else:
            if len(no_visited_idx_list) == 1:
                return None
            edge_idx = skip_idx
            while edge_idx == skip_idx:
                edge_idx = np.random.choice(no_visited_idx_list, 1)[0]

        return edge_idx

    def get_action_idx(self, action_probs):
        return self.model.get_action_idx(action_probs, self.temperature)

    def select(self):
        if not self.current_node.edges:
            return True
        edge_idx = None
        if self.current_node is self.root_node:
            edge_idx = self.choice_no_visited_edge_idx()
        if edge_idx is None:
            select_scores = np.array(
                [edge.get_select_score(self.current_node.edges, self.c_puct) for edge in self.current_node.edges])

            edge_idx = self.choice_edge_idx(select_scores)

            self.log(select_scores)

        is_repeat = self.env.check_repeat(self.current_node.edges[edge_idx].action, self.action_history)

        if is_repeat:
            if len(self.current_node.edges) == 1:
                return 2
            else:
                tmp_edge_idx = None
                if self.current_node is self.root_node:
                    tmp_edge_idx = self.choice_no_visited_edge_idx(edge_idx)
                if tmp_edge_idx is None:
                    select_scores = np.delete(select_scores, edge_idx, 0)
                    tmp_edge_idx = self.choice_edge_idx(select_scores)
                edge_idx = tmp_edge_idx

        self.selected_edges.append(self.current_node.edges[edge_idx])
        self.action_history.append(self.current_node.edges[edge_idx].action)
        self.state_history.append(self.current_node.edges[edge_idx].node.state)

        self.current_node = self.current_node.edges[edge_idx].node
        # self.env.print_env(state=self.current_node.state)

        return False

    def expand_and_evaluate(self):
        self.log("Expand and Evaluate!")

        if self.env.is_over(self.current_node.state):
            self.log("MCTS Game Over")

            return self.loser_reward

        # todo :pass액션 추가 ( 둘다 pass할경우 점수계산으로

        # action_probs, state_value = self.model.inference(
        #     common.convert_state_history_to_model_input(self.state_history[-(self.num_state_history + 1):],
        #                                                 self.num_state_history))
        state_value = 1
        action_probs = np.random.dirichlet([1] * 90, 1)[0]
        self.log("MCTS Value inference", state_value)
        # todo : <<빅장>> 혹은 외통수(장군)등 기능 구현?
        # todo: 비긴 상태 구현해서 적용하기(더 디테일하게)

        legal_actions = self.env.get_all_actions(self.current_node.state)

        if not legal_actions:
            return self.loser_reward
        legal_action_probs = self.model.filter_action_probs(action_probs, legal_actions, self.env)

        if self.root_node is self.current_node:
            # add noise to prior probabilities
            # if (legal_action_probs == 0).all():
            #     noise_probs = legal_action_probs
            # else:
            # noise_probs = np.random.dirichlet(legal_action_probs, 1)[0]
            noise_probs = np.random.dirichlet([1] * len(legal_action_probs), 1)[0]

            legal_action_probs = ((1 - 0.25) * legal_action_probs + (noise_probs * 0.25))

            legal_action_probs = legal_action_probs / legal_action_probs.sum()

        self.current_node.edges = []
        best_reward = 0
        for i, action_prob in enumerate(legal_action_probs):
            next_state, info = self.env.simulate(self.current_node.state, legal_actions[i])
            if info["reward"] > best_reward:
                best_reward = info["reward"]
            self.current_node.edges.append(
                Edge(self.current_node, action_prob, next_state, legal_actions[i], info["reward"]))
        # update reward
        tmp_node = self.current_node
        i = 0
        while tmp_node.parent_node is not None:
            if i > 0:
                best_reward = 0
                for edge in tmp_node.edges:
                    if edge.reward > best_reward:
                        best_reward = edge.reward
            if tmp_node.best_reward == best_reward:
                break
            diff_reward = best_reward - tmp_node.best_reward

            tmp_node.parent_edge.reward -= diff_reward
            tmp_node = tmp_node.parent_node
            i += 1

        # reward = -self.current_node.parent_edge.reward if self.current_node.parent_edge else 0
        #
        # state_value = 0.5 * state_value + 0.5 * reward

        self.log("MCTS state value + reward", state_value)
        return state_value

    def backup(self, state_value):
        self.log("MCTS Backup")

        self.selected_edges.reverse()

        for i, edge in enumerate(self.selected_edges):
            if i % 2 == 0:

                edge.update(-state_value)

            else:

                edge.update(state_value)

        self.init_state()

    def init_state(self):
        self.current_node = self.root_node
        self.state_history = [self.current_node.state]
        self.selected_edges = []
        self.action_history = []

    def print_tree(self):
        self.log("========== mcts tree trace ==========")

        self.print_row([self.root_node])

        self.log("=====================================")

    def print_row(self, nodes, row_idx=0):
        child_nodes = []
        for node in nodes:
            for edge in node.edges:
                child_nodes.append(edge.node)
        self.log("%d row: %d nodes" % (row_idx, len(nodes)))
        if row_idx > 996:
            self.log("more...")
            return
        if child_nodes:
            self.print_row(child_nodes, row_idx + 1)


class Node(object):
    def __init__(self, state, parent_edge=None, parent_node=None):
        self.state = state
        self.edges = []
        self.parent_edge = parent_edge
        self.parent_node = parent_node
        self.best_reward = .0


class Edge(object):
    def __init__(self, parent_node, action_prob, state, action, reward):
        # N
        self.visit_count = .0
        # W
        self.total_action_value = .0
        # Q
        self.mean_action_value = .0
        # P
        self.action_prob = action_prob
        self.action = action
        self.reward = reward
        self.node = Node(state, self, parent_node)

    def add_noise(self, noice_prob):

        self.action_prob = (0.75 * self.action_prob) + (0.25 * noice_prob)

    def update(self, state_value):
        self.visit_count += 1.
        self.total_action_value += state_value
        self.mean_action_value = self.total_action_value / self.visit_count

    def get_select_score(self, edges, c_puct):
        # todo : what is b?? other acitions visit count?? check!!

        total_other_edge_visit_count = .0
        for edge in edges:
            total_other_edge_visit_count += edge.visit_count
        U = c_puct * (math.sqrt(total_other_edge_visit_count) / (1. + self.visit_count))
        R = 0.5 * self.reward
        result = U + R
        # result = self.reward

        return result

    def get_action_probs(self, edges, temperature):
        # todo : what is b?? other acitions visit count?? check!!

        total_other_edge_visit_count = .0
        for edge in edges:
            if temperature == 0:
                total_other_edge_visit_count += edge.visit_count
            else:
                total_other_edge_visit_count += (pow(edge.visit_count, 1. / temperature))
        if temperature == 0:
            result = self.visit_count / total_other_edge_visit_count
        else:
            result = pow(self.visit_count, 1. / temperature) / total_other_edge_visit_count

        return result
