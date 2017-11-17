# coding=utf8
import math
import numpy as np
import random


class Mcts(object):
    def __init__(self, state, env, max_simulation=500, winner_reward=1, loser_reward=-1, c_puct=0.1, max_rollout=200):
        self.env = env
        self.max_simulation = max_simulation
        self.root_node = Node(state)
        self.prev_root_node = None
        self.selected_edges = []
        self.current_node = self.root_node
        self.root_turn = None
        self.current_turn = None
        self.temperature = 0
        self.winner_reward = winner_reward
        self.loser_reward = loser_reward
        self.c_puct = c_puct
        self.max_rollout = max_rollout

    def search(self, temperature=0, action_idx=None):
        self.temperature = temperature
        self.root_turn = self.env.current_turn
        self.current_turn = self.env.current_turn
        if action_idx:
            if not self.root_node.edges or len(self.root_node.edges) <= action_idx:
                self.expand_and_evaluate(False)
            self.root_node = self.root_node.edges[action_idx].node

        for i in range(self.max_simulation):
            print("mcts simulate %d " % i)
            self.simulate()

        action_probs = np.array(
            [edge.get_action_probs(self.root_node.edges, self.temperature) for edge in self.root_node.edges])
        if (action_probs == 0).all():
            action_idx = np.random.choice(range(len(action_probs)), 1)[0]
        else:
            arg_max_list = np.argwhere(action_probs == np.amax(action_probs)).flatten()
            print("MCTS Max score:%f" % arg_max_list[0])
            if len(arg_max_list) > 1:
                action_idx = np.random.choice(arg_max_list, 1)[0]
            else:
                action_idx = action_probs.argmax()
        searched_action = self.root_node.edges[action_idx].action
        print("MCTS Search Complete! visit count: %d, total_value: %f, mean_value: %f " % (
            self.root_node.edges[action_idx].visit_count, self.root_node.edges[action_idx].total_action_value,
            self.root_node.edges[action_idx].mean_action_value))
        self.prev_root_node = self.root_node
        self.root_node = self.root_node.edges[action_idx].node
        return searched_action

    def simulate(self):
        is_leaf_node = False
        i = 0
        while not is_leaf_node:
            print("mcts select %d" % i)
            is_leaf_node = self.select()
            i += 1

        state_value = self.expand_and_evaluate()
        self.backup(state_value)

    def select(self):
        if not self.current_node.edges:
            return True
        select_scores = np.array(
            [edge.get_select_score(self.current_node.edges, self.c_puct) for edge in self.current_node.edges])
        if (select_scores == 0).all():
            edge_idx = np.random.choice(range(len(select_scores)), 1)[0]
        else:
            arg_max_list = np.argwhere(select_scores == np.amax(select_scores)).flatten()
            if len(arg_max_list) > 1:
                edge_idx = np.random.choice(arg_max_list, 1)[0]
            else:
                edge_idx = select_scores.argmax()

        self.selected_edges.append(self.current_node.edges[edge_idx])
        self.current_node = self.current_node.edges[edge_idx].node
        self.env.print_env(state=self.current_node.state)
        self.current_turn = 'r' if self.current_turn == 'b' else 'b'

        return False

    def expand_and_evaluate(self, do_rollout=True):
        print("Expand and Evaluate!")
        if self.env.is_over(self.current_node.state):
            print("MCTS Game Over")
            return self.loser_reward

        state_value = 0
        if do_rollout:
            state_value = self.rollout(self.current_node.state)
            print("state_value %f" % state_value)

        legal_actions = self.env.get_all_actions(self.current_node.state)

        if not legal_actions:
            return self.loser_reward

        self.current_node.edges = [
            Edge(self.env.simulate(self.current_node.state, action, False), action) for
            i, action in enumerate(legal_actions)]

        return state_value

    def rollout(self, state):
        first_rewards = 0
        second_rewards = 0
        for i in range(self.max_rollout):
            legal_actions = self.env.get_all_actions(self.current_node.state)
            action = legal_actions[random.randint(0, len(legal_actions) - 1)]
            state, info = self.env.simulate(state, action)
            if info["is_game_over"]:
                if i % 2 == 0:
                    return self.winner_reward
                else:
                    return self.loser_reward
            else:
                if i % 2 == 0:
                    first_rewards += info["reward"]
                else:
                    second_rewards += info["reward"]
        return float(first_rewards - second_rewards) / 73 * self.winner_reward

    def backup(self, state_value):
        print("MCTS Backup")
        self.selected_edges.reverse()
        for i, edge in enumerate(self.selected_edges):
            if i % 2 == 0:
                state_value = -state_value
            edge.update(state_value)

        self.current_node = self.root_node
        self.selected_edges = []

    def print_tree(self):
        print("========== mcts tree trace ==========")
        self.print_row([self.prev_root_node])
        print("=====================================")

    def print_row(self, nodes, row_idx=0):
        child_nodes = []
        for node in nodes:
            for edge in node.edges:
                child_nodes.append(edge.node)
        print("%d row: %d nodes" % (row_idx, len(nodes)))
        if child_nodes:
            self.print_row(child_nodes, row_idx + 1)


class Node(object):
    def __init__(self, state):
        self.state = state
        self.edges = []


class Edge(object):
    def __init__(self, state, action):
        # N
        self.visit_count = 0
        # W
        self.total_action_value = 0
        # Q
        self.mean_action_value = 0
        # P
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
        U = c_puct * (math.sqrt(total_other_edge_visit_count) / (1 + self.visit_count))
        return self.mean_action_value + U

    def get_action_probs(self, edges, temperature):
        # todo : what is b?? other acitions visit count?? check!!
        total_other_edge_visit_count = 0
        for edge in edges:
            if temperature == 0:
                total_other_edge_visit_count += edge.visit_count
            else:
                total_other_edge_visit_count += (pow(edge.visit_count, 1 / temperature))
        if temperature == 0:
            return self.mean_action_value + self.visit_count / total_other_edge_visit_count
        else:
            return self.mean_action_value + pow(self.visit_count, 1 / temperature) / total_other_edge_visit_count
