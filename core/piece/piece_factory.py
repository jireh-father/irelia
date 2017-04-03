from core.constant import Constant
from core.piece import cannon
from core.piece import car
from core.piece import guardian
from core.piece import horse
from core.piece import king
from core.piece import sang
from core.piece import soldier


def get_actions(state_map, x, y, color=Constant.BLUE, is_hash_map=False):
    piece_num = int(state_map[y][x][1])
    if piece_num == Constant.KING:
        return king.get_actions(state_map, x, y, color, is_hash_map)
    elif piece_num == Constant.SOLDIER:
        return soldier.get_actions(state_map, x, y, color, is_hash_map)
    elif piece_num == Constant.SANG:
        return sang.get_actions(state_map, x, y, color, is_hash_map)
    elif piece_num == Constant.GUARDIAN:
        return guardian.get_actions(state_map, x, y, color, is_hash_map)
    elif piece_num == Constant.HORSE:
        return horse.get_actions(state_map, x, y, color, is_hash_map)
    elif piece_num == Constant.CANNON:
        return cannon.get_actions(state_map, x, y, color, is_hash_map)
    elif piece_num == Constant.CAR:
        return car.get_actions(state_map, x, y, color, is_hash_map)
