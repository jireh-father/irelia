import numpy as np
import random
import copy
import sys


class ActionSpace(object):
  n = None

  def __init__(self, n):
    self.n = n


class KoreanJanggiActionSpace(ActionSpace):
  default_action_space_n = 40

  def __init__(self):
    ActionSpace.__init__(self, KoreanJanggiActionSpace.default_action_space_n)


class Environment(object):
  properties = None
  action_space = None

  def __init__(self, properties):
    self.properties = properties
    self.action_space = KoreanJanggiActionSpace()

  def reset(self):
    return 2

  def step(self, action):
    return 2


class KoreanJanggi(Environment):
  KING = 46
  SOLDIER = 1
  SANG = 2
  GUARDIUN = 3
  HORSE = 4
  CANNON = 5
  CAR = 6

  state_list = {}
  rand_position_list = ['masangmasang', 'masangsangma', 'sangmasangma', 'sangmamasang']
  default_state_map = [
    [6, 0, 0, 3, 0, 3, 0, 0, 6],
    [0, 0, 0, 0, 46, 0, 0, 0, 0],
    [0, 5, 0, 0, 0, 0, 0, 5, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 5, 0, 0, 0, 0, 0, 5, 0],
    [0, 0, 0, 0, 46, 0, 0, 0, 0],
    [6, 0, 0, 3, 0, 3, 0, 0, 6],
  ]

  def __init__(self, properties, state_list={}):
    Environment.__init__(self, properties)
    self.state_list = state_list

  @staticmethod
  def convert_state_key(state_map):
    return str(state_map).replace('[', '').replace(', ', ',').replace(']', '')

  @staticmethod
  def get_available_actions(state_map):
    return [0] * 50

  def reset(self):
    if not self.properties['position_type'] or self.properties['position_type'] == 'random':
      before_rand_position = random.randint(0, 3)
      after_rand_position = random.randint(0, 3)
      position_type_list = [KoreanJanggi.rand_position_list[before_rand_position],
                            KoreanJanggi.rand_position_list[after_rand_position]]
    else:
      position_type_list = self.properties['position_type']

    default_map = copy.deepcopy(KoreanJanggi.default_state_map)

    for i, position_type in enumerate(position_type_list):
      if position_type == 'masangmasang':
        if i == 0:
          default_map[-1][1] = 4
          default_map[-1][2] = 2
          default_map[-1][6] = 4
          default_map[-1][7] = 2
        else:
          default_map[0][1] = 2
          default_map[0][2] = 4
          default_map[0][6] = 2
          default_map[0][7] = 4
      elif position_type == 'masangsangma':
        if i == 0:
          default_map[-1][1] = 4
          default_map[-1][2] = 2
          default_map[-1][6] = 2
          default_map[-1][7] = 4
        else:
          default_map[0][1] = 4
          default_map[0][2] = 2
          default_map[0][6] = 2
          default_map[0][7] = 4
      elif position_type == 'sangmasangma':
        if i == 0:
          default_map[-1][1] = 2
          default_map[-1][2] = 4
          default_map[-1][6] = 2
          default_map[-1][7] = 4
        else:
          default_map[0][1] = 4
          default_map[0][2] = 2
          default_map[0][6] = 4
          default_map[0][7] = 2
      elif position_type == 'sangmamasang':
        if i == 0:
          default_map[-1][1] = 2
          default_map[-1][2] = 4
          default_map[-1][6] = 4
          default_map[-1][7] = 2
        else:
          default_map[0][1] = 2
          default_map[0][2] = 4
          default_map[0][6] = 4
          default_map[0][7] = 2
      else:
        raise Exception('position_type is invalid : ' + position_type)
    state_key = KoreanJanggi.convert_state_key(default_map)
    if state_key not in self.state_list:
      self.state_list[state_key] = \
        {'state_map': default_map, 'action_list': KoreanJanggi.get_available_actions(default_map)}

    self.print_map(state_key)

    return state_key

  def print_map(self, state):
    for line in self.state_list[state]['state_map']:
      converted_line = []
      for val in line:
        if val == 0:
          converted_line.append('--')
        elif val == KoreanJanggi.SOLDIER:
          converted_line.append('SD')
        elif val == KoreanJanggi.SANG:
          converted_line.append('SG')
        elif val == KoreanJanggi.GUARDIUN:
          converted_line.append('GD')
        elif val == KoreanJanggi.HORSE:
          converted_line.append('HS')
        elif val == KoreanJanggi.CANNON:
          converted_line.append('CN')
        elif val == KoreanJanggi.CAR:
          converted_line.append('CR')
        elif val == KoreanJanggi.KING:
          converted_line.append('KG')
      print(converted_line)

  def step(self, action, state):
    self.print_map(state)
    return 2

  def get_action(self, Q, state, i, is_red=False):
    if is_red:
      # reverse state
      state
    if not Q or state not in Q:
      # if state is not in the Q, create state map and actions by state hash key
      Q[state] = np.zeros([len(self.state_list[state]['action_list'])])
    action_cnt = len(Q[state])
    return np.argmax(Q[state] + np.random.randn(1, action_cnt) / (i + 1))


class IrelGym(object):
  env_class_map = {'KoreanJanggi': KoreanJanggi}

  properties = {}

  @staticmethod
  def register(id, properties):
    IrelGym.properties[id] = properties

  @staticmethod
  def make(game_id):
    if not game_id:
      raise Exception('game id is not exist.')
    if game_id not in IrelGym.env_class_map:
      raise Exception('Unknown game_id.')
    return IrelGym.env_class_map[game_id](IrelGym.properties[game_id])

  def reset(self):
    return 2


def get_action(Q, state, action_space_cnt):
  return np.argmax(Q[state, :] + np.random.randn(1, action_space_cnt) / (i + 1))


IrelGym.register('KoreanJanggi', {'position_type': 'random'})

env = IrelGym.make('KoreanJanggi')
# load q table if existed.
Q_green = {}
Q_red = {}

dis = .99
num_episodes = 2000

green_reward_list = []
red_reward_list = []

for i in range(num_episodes):
  green_state = env.reset()
  green_reward_all = 0
  red_reward_all = 0
  green_done = False
  red_done = False

  while not green_done and not red_done:
    green_action = env.get_action(Q_green, green_state, i)
    red_state, green_reward, green_done = env.step(green_action, green_state)

    if old_red_state:
      Q_red[old_red_state, red_action] = (red_reward - green_reward) + dis * np.max(Q_red[red_state])
      red_reward_all += (red_reward - green_reward)

    red_action = env.get_action(Q_red, red_state, i)
    next_green_state, red_reward, red_done, _ = env.step(red_action, red_state)

    Q_green[green_state, green_action] = (green_reward - red_reward) + dis * np.max(Q_green[next_green_state])
    green_reward_all += (green_reward - red_reward)

    green_state = next_green_state
    old_red_state = red_state

  green_reward_list.append(green_reward_all)
  red_reward_list.append(red_reward_all)
