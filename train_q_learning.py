from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environment.gym import Gym

Gym.register('KoreanChess', {'position_type': 'random'})

env = Gym.make('KoreanChess')
# load q table if existed.
Q_blue = {}
Q_red = {}

dis = .99
num_episodes = 2000

blue_reward_list = []
red_reward_list = []

for i in range(num_episodes):
    blue_state = env.reset()
    blue_reward_all = 0
    red_reward_all = 0
    blue_done = False
    red_done = False

    while not blue_done and not red_done:
        blue_action = env.get_action(Q_blue, blue_state, i)
        red_state, blue_reward, blue_done = env.step(blue_action, blue_state)

        if old_red_state:
            Q_red[old_red_state][red_action] = (red_reward - blue_reward) + dis * np.max(Q_red[red_state])
            red_reward_all += (red_reward - blue_reward)

        red_action = env.get_action(Q_red, red_state, i, True)
        next_blue_state, red_reward, red_done, _ = env.step(red_action, red_state)

        Q_blue[blue_state][blue_action] = (blue_reward - red_reward) + dis * np.max(Q_blue[next_blue_state])
        blue_reward_all += (blue_reward - red_reward)

        blue_state = next_blue_state
        old_red_state = red_state

    blue_reward_list.append(blue_reward_all)
    red_reward_list.append(red_reward_all)
