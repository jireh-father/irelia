from agent.agent import Agent


class RandomAgent(Agent):
    def get_action(self, Q, state, i, is_red=False):
        action_list = self.get_action_list(state, is_red)
        action_cnt = len(action_list)
        if not Q or state not in Q:
            # if state is not in the Q, create state map and actions by state hash key
            Q[state] = np.zeros(action_cnt)

        if action_cnt < 1 or np.sum(Q[state]) == 0:
            q_state_key_list = {}
            for q_state_key in Q:
                diff_score = Core.compare_state(state, q_state_key)
                q_state_key_list[q_state_key] = diff_score

            sorted_q_state_list = sorted(q_state_key_list.items(), key=operator.itemgetter(1))
            for item in sorted_q_state_list:
                q_state = item[0]
                if np.sum(Q[q_state]) == 0:
                    continue
                q_max_action_no = np.argmax(Q[q_state])
                q_action_list = self.get_action_list(q_state, is_red)
                q_action = q_action_list[q_max_action_no]
                for i, action in enumerate(action_list):
                    if action['x'] == q_action['x'] \
                      and action['y'] == q_action['y'] \
                      and action['to_x'] == q_action['to_x'] \
                      and action['to_y'] == q_action['to_y']:
                        return i

        return np.argmax(Q[state] + np.random.randn(1, action_cnt) / (action_cnt * 10))
