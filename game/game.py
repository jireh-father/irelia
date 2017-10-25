from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from game.korean_chess_v1 import KoreanChessV1

class Game(object):
    env_class_map = {'KoreanChess-v1': KoreanChessV1}

    @staticmethod
    def make(game_id, properties):
        if not game_id:
            raise Exception('game id is not exist.')
        if game_id not in Game.env_class_map:
            raise Exception('Unknown game id.')
        return Game.env_class_map[game_id](properties)

