from collections import namedtuple
# 这个模块实现了特定目标的容器，以提供Python标准内建容器 dict , list , set , 和 tuple 的替代选择。

Point = namedtuple('Point', 'X Y')

offset = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 棋盘的4个方向


class get_score:
    def __init__(self, board_line, my_value, opponent_value):
        self.board_line = board_line
        self.my_value = my_value   # AI棋子属性(白色)
        self.opponent_value = opponent_value   # 对手的棋子属性

    def all_score(self, check_board):
        score = 0
        for i in range(self.board_line):
            for j in range(self.board_line):
                # 计算每一个空子点的得分
                if check_board[j][i] == 0:
                    _score = self.get_point_score(Point(i, j), check_board)
                    score += _score   # 计算当前棋局的总得分
        return score   # 返回AI落子位置

    def get_point_score(self, point, check_board):
        score = 0
        for os in offset:    # 四个方向
            score += self.get_direction_score(point, os[0], os[1], check_board)    # 计算四个方向得分
        return score

    def get_direction_score(self, point, x_offset, y_offset, check_board):
        count = 0  # 落子处我方连续子数
        _count = 0  # 落子处对方连续子数
        space = None  # 我方连续子中有无空格
        _space = None  # 对方连续子中有无空格
        both = 0  # 我方连续子两端有无阻挡
        _both = 0  # 对方连续子两端有无阻挡

        # 如果是 1 表示是边上是我方子，2 表示敌方子
        flag = self._get_stone_color(point, x_offset, y_offset, True, check_board)
        if flag != 0:
            for step in range(1, 6):   # 在一个方向上走的步数
                x = point.X + step * x_offset
                y = point.Y + step * y_offset

                # 是否在该方向上落子在棋盘内
                if 0 <= x < self.board_line and 0 <= y < self.board_line:
                    if flag == 1:
                        if check_board[y][x] == self.my_value:
                            count += 1
                            if space is False:
                                space = True
                        elif check_board[y][x] == self.opponent_value:
                            _both += 1
                            break
                        else:
                            if space is None:
                                space = False
                            else:
                                break  # 遇到第二个空格退出
                    elif flag == 2:
                        if check_board[y][x] == self.my_value:
                            _both += 1
                            break
                        elif check_board[y][x] == self.opponent_value:
                            _count += 1
                            if _space is False:
                                _space = True
                        else:
                            if _space is None:
                                _space = False
                            else:
                                break
                else:
                    # 遇到边也就是阻挡
                    if flag == 1:
                        both += 1
                    elif flag == 2:
                        _both += 1

        if space is False:
            space = None
        if _space is False:
            _space = None

        # 计算当前方向的相反方向
        _flag = self._get_stone_color(point, -x_offset, -y_offset, True, check_board)
        if _flag != 0:
            for step in range(1, 6):
                x = point.X - step * x_offset
                y = point.Y - step * y_offset
                if 0 <= x < self.board_line and 0 <= y < self.board_line:
                    if _flag == 1:
                        if check_board[y][x] == self.my_value:
                            count += 1
                            if space is False:
                                space = True
                        elif check_board[y][x] == self.opponent_value:
                            _both += 1
                            break
                        else:
                            if space is None:
                                space = False
                            else:
                                break  # 遇到第二个空格退出
                    elif _flag == 2:
                        if check_board[y][x] == self.my_value:
                            _both += 1
                            break
                        elif check_board[y][x] == self.opponent_value:
                            _count += 1
                            if _space is False:
                                _space = True
                        else:
                            if _space is None:
                                _space = False
                            else:
                                break
                else:
                    # 遇到边也就是阻挡
                    if _flag == 1:
                        both += 1
                    elif _flag == 2:
                        _both += 1

        # 根据情况计算得分
        score = 0
        if count == 4:   # 落子处我方连续数
            score += 10
        if _count == 4:    # 落子处对方连续数
            score -= 15
        if count == 3:
            if both == 0:   # 我方连续子无阻挡
                score += 5
            elif both == 1:   # 我方连续子有阻挡
                score += 3
            else:
                score += 0
        if _count == 3:
            if _both == 0:   # 对方连续子无阻挡
                score -= 5
            elif _both == 1:   # 对方连续子有阻挡
                score += 3
            else:
                score += 0
        if count == 2:
            if both == 0:
                score += 1
            elif both == 1:
                score += 0.5
            else:
                score += 0
        if _count == 2:
            if _both == 0:
                score -= 1
            elif _both == 1:
                score -= 0.5
            else:
                score = 0
        if space or _space:
            score /= 2

        return score   # 返回落子得分

    # 判断指定位置处在指定方向上是我方子、对方子、空
    def _get_stone_color(self, point, x_offset, y_offset, next, check_board):
        # 计算当前位置的指定方向上的位置
        x = point.X + x_offset
        y = point.Y + y_offset

        # 判断指定位置是否在棋盘内
        if 0 <= x < self.board_line and 0 <= y < self.board_line:
            if check_board[y][x] == self.my_value:   # 指定位置为AI子
                return 1
            elif check_board[y][x] == self.opponent_value:   # 指定位置为对手子
                return 2
            else:
                if next:
                    return self._get_stone_color(Point(x, y), x_offset, y_offset, False, check_board)
                else:
                    return 0
        else:
            return 0


"""
model = get_score(board_line=15, my_value=2, opponent_value=1)

data = [0 for _ in range(15)]

shuju = [data for _ in range(15)]

print(shuju)


out = model.all_score(shuju)
print(out)
"""