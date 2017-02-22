from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from environment.korean_chess import KoreanChess

class Gym(object):
    env_class_map = {'KoreanChess': KoreanChess}

    properties = {}

    @staticmethod
    def register(id, properties):
        Gym.properties[id] = properties

    @staticmethod
    def make(game_id):
        if not game_id:
            raise Exception('game id is not exist.')
        if game_id not in Gym.env_class_map:
            raise Exception('Unknown game_id.')
        return Gym.env_class_map[game_id](Gym.properties[game_id])

    def reset(self):
        return 2