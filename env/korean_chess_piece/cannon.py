from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from env import korean_chess_util as kcu


def get_actions(state_map, x, y):
    piece = state_map[y][x]

    side = piece[0]

    action_list = []

    piece_type = kcu.CANNON

    # 대각선 오른쪽 전진 길 체크
    if (y == 9 and x == 3) or (y == 2 and x == 3):
        # 왕자리에 디딤돌이 있는지 체크 (포가아니면서 빈자리가 아니면됨)

        step_stone_x = x + 1
        step_stone_y = y - 1
        if kcu.is_piece(state_map, step_stone_x, step_stone_y) \
          and not kcu.is_cannon(state_map, step_stone_x, step_stone_y):
            # 디딤돌 다음 자리에 상대편이나(포가 아니면서) 빈자리가 있는지 체크
            next_of_step_stone_x = step_stone_x + 1
            next_of_step_stone_y = step_stone_y - 1
            if kcu.is_empty_space(state_map, next_of_step_stone_x, next_of_step_stone_y) \
              or kcu.is_enemy(state_map, next_of_step_stone_x, next_of_step_stone_y):
                action_list.append(
                    {'x': x, 'y': y, 'direction': kcu.DIAGONAL_FORWARD_RIGHT1, 'step': 2,
                     'piece_type': piece_type})

    # 대각선 왼쪽 전진 길 체크
    if (y == 9 and x == 5) or (y == 2 and x == 5):
        # 왕자리에 디딤돌이 있는지 체크 (포가아니면서 빈자리가 아니면됨)
        step_stone_x = x - 1
        step_stone_y = y - 1
        if kcu.is_piece(state_map, step_stone_x, step_stone_y) \
          and not kcu.is_cannon(state_map, step_stone_x, step_stone_y):
            # 디딤돌 다음 자리에 상대편이나(포가 아니면서) 빈자리가 있는지 체크
            next_of_step_stone_x = step_stone_x - 1
            next_of_step_stone_y = step_stone_y - 1
            if kcu.is_empty_space(state_map, next_of_step_stone_x, next_of_step_stone_y) \
              or kcu.is_enemy(state_map, next_of_step_stone_x, next_of_step_stone_y):
                action_list.append(
                    {'x': x, 'y': y, 'direction': kcu.DIAGONAL_FORWARD_LEFT1, 'step': 2,
                     'piece_type': piece_type})

    # 대각선 오른쪽 후진 길 체크
    if (y == 7 and x == 3) or (y == 0 and x == 3):
        # 왕자리에 디딤돌이 있는지 체크 (포가아니면서 빈자리가 아니면됨)
        step_stone_x = x + 1
        step_stone_y = y + 1
        if kcu.is_piece(state_map, step_stone_x, step_stone_y) \
          and not kcu.is_cannon(state_map, step_stone_x, step_stone_y):
            # 디딤돌 다음 자리에 상대편이나(포가 아니면서) 빈자리가 있는지 체크
            next_of_step_stone_x = step_stone_x + 1
            next_of_step_stone_y = step_stone_y + 1
            if kcu.is_empty_space(state_map, next_of_step_stone_x, next_of_step_stone_y) \
              or kcu.is_enemy(state_map, next_of_step_stone_x, next_of_step_stone_y):
                action_list.append(
                    {'x': x, 'y': y, 'direction': kcu.DIAGONAL_BACKWARD_RIGHT1, 'step': 2,
                     'piece_type': piece_type})

    # 대각선 왼쪽 후진 길 체크
    if (y == 7 and x == 5) or (y == 0 and x == 5):
        # 왕자리에 디딤돌이 있는지 체크 (포가아니면서 빈자리가 아니면됨)
        step_stone_x = x - 1
        step_stone_y = y + 1
        if kcu.is_piece(state_map, step_stone_x, step_stone_y) \
          and not kcu.is_cannon(state_map, step_stone_x, step_stone_y):
            # 디딤돌 다음 자리에 상대편이나(포가 아니면서) 빈자리가 있는지 체크
            next_of_step_stone_x = step_stone_x - 1
            next_of_step_stone_y = step_stone_y + 1
            if kcu.is_empty_space(state_map, next_of_step_stone_x, next_of_step_stone_y) \
              or kcu.is_enemy(state_map, next_of_step_stone_x, next_of_step_stone_y):
                action_list.append(
                    {'x': x, 'y': y, 'direction': kcu.DIAGONAL_BACKWARD_LEFT1, 'step': 2,
                     'piece_type': piece_type})

    # 전진 길 체크
    if y > kcu.TOP_WALL + 1:
        moving_y = y - 1
        step_stone_y = 0
        while moving_y > kcu.TOP_WALL:
            # 건널수 있는게 있는지 검색
            # 가장먼저 포가 나오면 실패
            if kcu.is_cannon(state_map, x, moving_y):
                break

            if kcu.is_piece(state_map, x, moving_y) and not kcu.is_cannon(state_map, x, moving_y):
                step_stone_y = moving_y
                break
            moving_y -= 1

        # 포가아닌 말의 한칸후부터 포이거나 우리편말이나오기 전칸까지 액션 추가
        if step_stone_y > kcu.TOP_WALL and step_stone_y < y:
            next_of_step_stone_y = step_stone_y - 1
            step = 1
            add_step = y - step_stone_y
            while next_of_step_stone_y >= kcu.TOP_WALL:
                # 포이거나 우리편이면 정지
                if kcu.is_cannon(state_map, x, next_of_step_stone_y) or \
                  kcu.is_our_side(state_map, x, next_of_step_stone_y, side):
                    break

                action_list.append(
                    {'x': x, 'y': y, 'direction': kcu.FORWARD, 'step': step + add_step, 'piece_type': piece_type})

                if kcu.is_enemy(state_map, x, next_of_step_stone_y, side):
                    break

                next_of_step_stone_y -= 1
                step += 1

    # 오른쪽 길 체크
    if x < kcu.RIGHT_WALL - 1:
        moving_x = x + 1
        step_stone_x = 0
        while moving_x < kcu.RIGHT_WALL:
            # 건널수 있는게 있는지 검색
            # 가장먼저 포가 나오면 실패
            if kcu.is_cannon(state_map, moving_x, y):
                break

            if kcu.is_piece(state_map, moving_x, y) and not kcu.is_cannon(state_map, moving_x, y):
                step_stone_x = moving_x
                break
            moving_x += 1

        # 포가아닌 말의 한칸후부터 포이거나 우리편말이나오기 전칸까지 액션 추가
        if step_stone_x < kcu.RIGHT_WALL and step_stone_x > x:
            next_of_step_stone_x = step_stone_x + 1
            step = 1
            add_step = step_stone_x - x
            while next_of_step_stone_x <= kcu.RIGHT_WALL:
                # 포이거나 우리편이면 정지
                if kcu.is_cannon(state_map, next_of_step_stone_x, y) or \
                  kcu.is_our_side(state_map, next_of_step_stone_x, y, side):
                    break

                action_list.append(
                    {'x': x, 'y': y, 'direction': kcu.RIGHT, 'step': step + add_step, 'piece_type': piece_type})

                if kcu.is_enemy(state_map, next_of_step_stone_x, y, side):
                    break

                next_of_step_stone_x += 1
                step += 1

    # 왼쪽 길 체크
    if x > kcu.LEFT_WALL + 1:
        moving_x = x - 1
        step_stone_x = 0
        while moving_x > kcu.LEFT_WALL:
            # 건널수 있는게 있는지 검색
            # 가장먼저 포가 나오면 실패
            if kcu.is_cannon(state_map, moving_x, y):
                break

            if kcu.is_piece(state_map, moving_x, y) and not kcu.is_cannon(state_map, moving_x, y):
                step_stone_x = moving_x
                break
            moving_x -= 1

        # 포가아닌 말의 한칸후부터 포이거나 우리편말이나오기 전칸까지 액션 추가
        if step_stone_x > kcu.LEFT_WALL and step_stone_x < x:
            next_of_step_stone_x = step_stone_x - 1
            step = 1
            add_step = x - step_stone_x
            while next_of_step_stone_x >= kcu.LEFT_WALL:
                # 포이거나 우리편이면 정지
                if kcu.is_cannon(state_map, next_of_step_stone_x, y) or \
                  kcu.is_our_side(state_map, next_of_step_stone_x, y, side):
                    break

                action_list.append(
                    {'x': x, 'y': y, 'direction': kcu.LEFT, 'step': step + add_step, 'piece_type': piece_type})

                if kcu.is_enemy(state_map, next_of_step_stone_x, y, side):
                    break

                next_of_step_stone_x -= 1
                step += 1

    # 후진 길 체크
    if y < kcu.BOTTOM_WALL - 1:
        moving_y = y + 1
        step_stone_y = 0
        while moving_y < kcu.BOTTOM_WALL:
            # 건널수 있는게 있는지 검색
            # 가장먼저 포가 나오면 실패
            if kcu.is_cannon(state_map, x, moving_y):
                break

            if kcu.is_piece(state_map, x, moving_y) and not kcu.is_cannon(state_map, x, moving_y):
                step_stone_y = moving_y
                break
            moving_y += 1

        # 포가아닌 말의 한칸후부터 포이거나 우리편말이나오기 전칸까지 액션 추가
        if step_stone_y < kcu.BOTTOM_WALL and step_stone_y > y:
            next_of_step_stone_y = step_stone_y - 1
            step = 1
            add_step = step_stone_y - y
            while next_of_step_stone_y <= kcu.BOTTOM_WALL:
                # 포이거나 우리편이면 정지
                if kcu.is_cannon(state_map, x, next_of_step_stone_y) or \
                  kcu.is_our_side(state_map, x, next_of_step_stone_y, side):
                    break

                action_list.append(
                    {'x': x, 'y': y, 'direction': kcu.BACKWARD, 'step': step + add_step, 'piece_type': piece_type})

                if kcu.is_enemy(state_map, x, next_of_step_stone_y, side):
                    break

                next_of_step_stone_y += 1
                step += 1

    return action_list
