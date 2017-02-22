from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Environment(object):
    properties = None
    action_space = None

    def __init__(self, properties):
        self.properties = properties
        self.action_space = KoreanChessActionSpace()

    def reset(self):
        return 2

    def step(self, action):
        return 2