from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Env(object):
    properties = None

    def __init__(self, properties):
        self.properties = properties

    def reset(self):
        pass

    def step(self, action, state_key, is_red=False):
        pass
