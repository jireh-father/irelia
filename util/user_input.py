from builtins import input


def get_user_input():
    while True:
        from_x = input("from X number : ")
        if from_x.isdigit():
            from_x = int(from_x)
            break
    while True:
        from_y = input("from Y number : ")
        if from_y.isdigit():
            from_y = int(from_y)
            break
    while True:
        to_x = input("to X number : ")
        if to_x.isdigit():
            to_x = int(to_x)
            break
    while True:
        to_y = input("to Y number : ")
        if to_y.isdigit():
            to_y = int(to_y)
            break
    return from_x, from_y, to_x, to_y
