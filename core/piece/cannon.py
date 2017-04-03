# coding=utf8
from core.constant import Constant
from core import game
from core.piece import common


def get_actions(state_map, x, y, color=Constant.BLUE, is_hash_map=False):
    piece = state_map[y][x]

    side = piece[0]

    if color is Constant.BLUE:
        action_list = []
    else:
        action_list = {}

    piece_type = Constant.CANNON

    # 대각선 오른쪽 전진 길 체크
    if (y == 9 and x == 3) or (y == 2 and x == 3):
        # 왕자리에 디딤돌이 있는지 체크 (포가아니면서 빈자리가 아니면됨)

        step_stone_x = x + 1
        step_stone_y = y - 1
        if game.is_piece(state_map, step_stone_x, step_stone_y) \
          and not game.is_cannon(state_map, step_stone_x, step_stone_y):
            # 디딤돌 다음 자리에 상대편이나(포가 아니면서) 빈자리가 있는지 체크
            next_of_step_stone_x = step_stone_x + 1
            next_of_step_stone_y = step_stone_y - 1
            if game.is_empty_space(state_map, next_of_step_stone_x, next_of_step_stone_y) \
              or game.is_enemy(state_map, next_of_step_stone_x, next_of_step_stone_y, side):
                common.add_action_to_list(x, y, next_of_step_stone_x, next_of_step_stone_y, action_list, color,
                                          is_hash_map)

    # 대각선 왼쪽 전진 길 체크
    if (y == 9 and x == 5) or (y == 2 and x == 5):
        # 왕자리에 디딤돌이 있는지 체크 (포가아니면서 빈자리가 아니면됨)
        step_stone_x = x - 1
        step_stone_y = y - 1
        if game.is_piece(state_map, step_stone_x, step_stone_y) \
          and not game.is_cannon(state_map, step_stone_x, step_stone_y):
            # 디딤돌 다음 자리에 상대편이나(포가 아니면서) 빈자리가 있는지 체크
            next_of_step_stone_x = step_stone_x - 1
            next_of_step_stone_y = step_stone_y - 1
            if game.is_empty_space(state_map, next_of_step_stone_x, next_of_step_stone_y) \
              or game.is_enemy(state_map, next_of_step_stone_x, next_of_step_stone_y, side):
                common.add_action_to_list(x, y, next_of_step_stone_x, next_of_step_stone_y, action_list, color,
                                          is_hash_map)

    # 대각선 오른쪽 후진 길 체크
    if (y == 7 and x == 3) or (y == 0 and x == 3):
        # 왕자리에 디딤돌이 있는지 체크 (포가아니면서 빈자리가 아니면됨)
        step_stone_x = x + 1
        step_stone_y = y + 1
        if game.is_piece(state_map, step_stone_x, step_stone_y) \
          and not game.is_cannon(state_map, step_stone_x, step_stone_y):
            # 디딤돌 다음 자리에 상대편이나(포가 아니면서) 빈자리가 있는지 체크
            next_of_step_stone_x = step_stone_x + 1
            next_of_step_stone_y = step_stone_y + 1
            if game.is_empty_space(state_map, next_of_step_stone_x, next_of_step_stone_y) \
              or game.is_enemy(state_map, next_of_step_stone_x, next_of_step_stone_y, side):
                common.add_action_to_list(x, y, next_of_step_stone_x, next_of_step_stone_y, action_list, color,
                                          is_hash_map)

    # 대각선 왼쪽 후진 길 체크
    if (y == 7 and x == 5) or (y == 0 and x == 5):
        # 왕자리에 디딤돌이 있는지 체크 (포가아니면서 빈자리가 아니면됨)
        step_stone_x = x - 1
        step_stone_y = y + 1
        if game.is_piece(state_map, step_stone_x, step_stone_y) \
          and not game.is_cannon(state_map, step_stone_x, step_stone_y):
            # 디딤돌 다음 자리에 상대편이나(포가 아니면서) 빈자리가 있는지 체크
            next_of_step_stone_x = step_stone_x - 1
            next_of_step_stone_y = step_stone_y + 1
            if game.is_empty_space(state_map, next_of_step_stone_x, next_of_step_stone_y) \
              or game.is_enemy(state_map, next_of_step_stone_x, next_of_step_stone_y, side):
                common.add_action_to_list(x, y, next_of_step_stone_x, next_of_step_stone_y, action_list, color,
                                          is_hash_map)

    # 전진 길 체크
    if y > Constant.TOP_WALL + 1:
        moving_y = y - 1
        step_stone_y = 0
        while moving_y > Constant.TOP_WALL:
            # 건널수 있는게 있는지 검색
            # 가장먼저 포가 나오면 실패
            if game.is_cannon(state_map, x, moving_y):
                break

            if game.is_piece(state_map, x, moving_y) and not game.is_cannon(state_map, x, moving_y):
                step_stone_y = moving_y
                break
            moving_y -= 1

        # 포가아닌 말의 한칸후부터 포이거나 우리편말이나오기 전칸까지 액션 추가
        if step_stone_y > Constant.TOP_WALL and step_stone_y < y:
            next_of_step_stone_y = step_stone_y - 1
            step = 1
            add_step = y - step_stone_y
            while next_of_step_stone_y >= Constant.TOP_WALL:
                # 포이거나 우리편이면 정지
                if game.is_cannon(state_map, x, next_of_step_stone_y) or \
                  game.is_our_side(state_map, x, next_of_step_stone_y, side):
                    break
                common.add_action_to_list(x, y, x, next_of_step_stone_y, action_list, color, is_hash_map)
                if game.is_enemy(state_map, x, next_of_step_stone_y, side):
                    break

                next_of_step_stone_y -= 1
                step += 1

    # 오른쪽 길 체크
    if x < Constant.RIGHT_WALL - 1:
        moving_x = x + 1
        step_stone_x = 0
        while moving_x < Constant.RIGHT_WALL:
            # 건널수 있는게 있는지 검색
            # 가장먼저 포가 나오면 실패
            if game.is_cannon(state_map, moving_x, y):
                break

            if game.is_piece(state_map, moving_x, y) and not game.is_cannon(state_map, moving_x, y):
                step_stone_x = moving_x
                break
            moving_x += 1

        # 포가아닌 말의 한칸후부터 포이거나 우리편말이나오기 전칸까지 액션 추가
        if step_stone_x < Constant.RIGHT_WALL and step_stone_x > x:
            next_of_step_stone_x = step_stone_x + 1
            step = 1
            add_step = step_stone_x - x
            while next_of_step_stone_x <= Constant.RIGHT_WALL:
                # 포이거나 우리편이면 정지
                if game.is_cannon(state_map, next_of_step_stone_x, y) or \
                  game.is_our_side(state_map, next_of_step_stone_x, y, side):
                    break
                common.add_action_to_list(x, y, next_of_step_stone_x, y, action_list, color, is_hash_map)

                if game.is_enemy(state_map, next_of_step_stone_x, y, side):
                    break

                next_of_step_stone_x += 1
                step += 1

    # 왼쪽 길 체크
    if x > Constant.LEFT_WALL + 1:
        moving_x = x - 1
        step_stone_x = 0
        while moving_x > Constant.LEFT_WALL:
            # 건널수 있는게 있는지 검색
            # 가장먼저 포가 나오면 실패
            if game.is_cannon(state_map, moving_x, y):
                break

            if game.is_piece(state_map, moving_x, y) and not game.is_cannon(state_map, moving_x, y):
                step_stone_x = moving_x
                break
            moving_x -= 1

        # 포가아닌 말의 한칸후부터 포이거나 우리편말이나오기 전칸까지 액션 추가
        if step_stone_x > Constant.LEFT_WALL and step_stone_x < x:
            next_of_step_stone_x = step_stone_x - 1
            step = 1
            add_step = x - step_stone_x
            while next_of_step_stone_x >= Constant.LEFT_WALL:
                # 포이거나 우리편이면 정지
                if game.is_cannon(state_map, next_of_step_stone_x, y) or \
                  game.is_our_side(state_map, next_of_step_stone_x, y, side):
                    break

                common.add_action_to_list(x, y, next_of_step_stone_x, y, action_list, color, is_hash_map)

                if game.is_enemy(state_map, next_of_step_stone_x, y, side):
                    break

                next_of_step_stone_x -= 1
                step += 1

    # 후진 길 체크
    if y < Constant.BOTTOM_WALL - 1:
        moving_y = y + 1
        step_stone_y = 0
        while moving_y < Constant.BOTTOM_WALL:
            # 건널수 있는게 있는지 검색
            # 가장먼저 포가 나오면 실패
            if game.is_cannon(state_map, x, moving_y):
                break

            if game.is_piece(state_map, x, moving_y) and not game.is_cannon(state_map, x, moving_y):
                step_stone_y = moving_y
                break
            moving_y += 1

        # 포가아닌 말의 한칸후부터 포이거나 우리편말이나오기 전칸까지 액션 추가
        if step_stone_y < Constant.BOTTOM_WALL and step_stone_y > y:
            next_of_step_stone_y = step_stone_y + 1
            step = 1
            add_step = step_stone_y - y
            while next_of_step_stone_y <= Constant.BOTTOM_WALL:
                # 포이거나 우리편이면 정지
                if game.is_cannon(state_map, x, next_of_step_stone_y) or \
                  game.is_our_side(state_map, x, next_of_step_stone_y, side):
                    break

                common.add_action_to_list(x, y, x, next_of_step_stone_y, action_list, color, is_hash_map)

                if game.is_enemy(state_map, x, next_of_step_stone_y, side):
                    break

                next_of_step_stone_y += 1
                step += 1

    return action_list
