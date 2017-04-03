from core.constant import Constant


def add_action_to_list(x, y, to_x, to_y, list, color, is_hash_map):
    if color is Constant.RED:
        x = 8 - x
        y = 9 - y
        to_x = 8 - to_x
        to_y = 9 - to_y
    action = {'x': x, 'y': y, 'to_x': to_x, 'to_y': to_y}
    if is_hash_map:
        list["%d_%d_%d_%d" % (x, y, to_x, to_y)] = action
    else:
        list.append(action)
