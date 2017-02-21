import numpy as np
import random
import copy


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

  def step(self):
    return 2


class KoreanJanggi(Environment):
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
    return 1

  def reset(self):
    if self.properties['position_type'] == 'random':
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
    if self.state_list and state_key not in self.state_list:
      self.state_list[state_key] = \
        {'state_map': default_map, 'action_list': KoreanJanggi.get_available_actions(default_map)}
    return state_key


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
Q1 = {}
Q2 = {}

dis = .99
num_episodes = 2000

beforeRewardList = []
afterRewardList = []

for i in range(num_episodes):
  after_state = env.reset()
  beforeRewardAll = 0
  afterRewardAll = 0
  before_done = False
  after_done = False

  while not before_done and not after_done:
    before_action = get_action(Q1, after_state, env.action_space.n)

    before_state, before_reward, before_done, _ = env.step(before_action)

    if after_action:
      afterRewardAll += (after_reward - before_reward)
      Q2[after_state, after_action] = (after_reward - before_reward) + dis * np.max(Q2[before_state, :])

    after_action = np.argmax(Q2[before_state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

    after_state, after_reward, after_done, _ = env.step(after_action)
    before_reward = before_reward - after_reward
    beforeRewardAll += before_reward
    Q1[before_state, before_action] = before_reward + dis * np.max(Q1[after_state, :])

  beforeRewardList.append(beforeRewardAll)
  afterRewardList.append(afterRewardAll)
