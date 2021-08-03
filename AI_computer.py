import random
from checkerboard import BLACK_CHESSMAN, WHITE_CHESSMAN, offset, Point


class AI:
    def __init__(self, line_points, chessman):
        self._line_points = line_points  # 棋盘大小
        self._my = chessman   # AI棋子属性(白色)
        self._opponent = BLACK_CHESSMAN if chessman == WHITE_CHESSMAN else WHITE_CHESSMAN  # 对手棋子属性
        self._checkerboard = [[0] * line_points for _ in range(line_points)]   # 棋盘初始化

    def get_opponent_drop(self, point):
        self._checkerboard[point.Y][point.X] = self._opponent.Value

    # AI落子位置
    def AI_drop(self):
        point = None
        score = 0
        for i in range(self._line_points):
            for j in range(self._line_points):
                # 计算每一个空子点的得分
                if self._checkerboard[j][i] == 0:
                    _score = self._get_point_score(Point(i, j))
                    if _score > score:    # 如果当前点得分更高则选择当前点位置
                        score = _score
                        point = Point(i, j)
                    elif _score == score and _score > 0:   # 如果得分相等则随机落子
                        r = random.randint(0, 100)
                        if r % 2 == 0:    # 一般的几率在得分相等时改变落子位置
                            point = Point(i, j)
        self._checkerboard[point.Y][point.X] = self._my.Value   # AI落子战格符 2
        return point   # 返回AI落子位置

    # 计算当前点得分
    def _get_point_score(self, point):
        score = 0
        for os in offset:    # 四个方向
            score += self._get_direction_score(point, os[0], os[1])    # 计算四个方向得分
        return score

    # 计算方向得分
    def _get_direction_score(self, point, x_offset, y_offset):
        count = 0  # 落子处我方连续子数
        _count = 0  # 落子处对方连续子数
        space = None  # 我方连续子中有无空格
        _space = None  # 对方连续子中有无空格
        both = 0  # 我方连续子两端有无阻挡
        _both = 0  # 对方连续子两端有无阻挡

        # 如果是 1 表示是边上是我方子，2 表示敌方子
        flag = self._get_stone_color(point, x_offset, y_offset, True)
        if flag != 0:
            for step in range(1, 6):   # 在一个方向上走的步数
                x = point.X + step * x_offset
                y = point.Y + step * y_offset

                # 是否在该方向上落子在棋盘内
                if 0 <= x < self._line_points and 0 <= y < self._line_points:
                    if flag == 1:
                        if self._checkerboard[y][x] == self._my.Value:
                            count += 1
                            if space is False:
                                space = True
                        elif self._checkerboard[y][x] == self._opponent.Value:
                            _both += 1
                            break
                        else:
                            if space is None:
                                space = False
                            else:
                                break  # 遇到第二个空格退出
                    elif flag == 2:
                        if self._checkerboard[y][x] == self._my.Value:
                            _both += 1
                            break
                        elif self._checkerboard[y][x] == self._opponent.Value:
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
        _flag = self._get_stone_color(point, -x_offset, -y_offset, True)
        if _flag != 0:
            for step in range(1, 6):
                x = point.X - step * x_offset
                y = point.Y - step * y_offset
                if 0 <= x < self._line_points and 0 <= y < self._line_points:
                    if _flag == 1:
                        if self._checkerboard[y][x] == self._my.Value:
                            count += 1
                            if space is False:
                                space = True
                        elif self._checkerboard[y][x] == self._opponent.Value:
                            _both += 1
                            break
                        else:
                            if space is None:
                                space = False
                            else:
                                break  # 遇到第二个空格退出
                    elif _flag == 2:
                        if self._checkerboard[y][x] == self._my.Value:
                            _both += 1
                            break
                        elif self._checkerboard[y][x] == self._opponent.Value:
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
            score = 10000
        elif _count == 4:    # 落子处对方连续数
            score = 9000
        elif count == 3:
            if both == 0:
                score = 1000
            elif both == 1:
                score = 100
            else:
                score = 0
        elif _count == 3:
            if _both == 0:
                score = 900
            elif _both == 1:
                score = 90
            else:
                score = 0
        elif count == 2:
            if both == 0:
                score = 100
            elif both == 1:
                score = 10
            else:
                score = 0
        elif _count == 2:
            if _both == 0:
                score = 90
            elif _both == 1:
                score = 9
            else:
                score = 0
        elif count == 1:
            score = 10
        elif _count == 1:
            score = 9
        else:
            score = 0

        if space or _space:
            score /= 2

        return score   # 返回落子得分

    # 判断指定位置处在指定方向上是我方子、对方子、空
    def _get_stone_color(self, point, x_offset, y_offset, next):
        # 计算当前位置的指定方向上的位置
        x = point.X + x_offset
        y = point.Y + y_offset

        # 判断指定位置是否在棋盘内
        if 0 <= x < self._line_points and 0 <= y < self._line_points:
            if self._checkerboard[y][x] == self._my.Value:   # 指定位置为AI子
                return 1
            elif self._checkerboard[y][x] == self._opponent.Value:   # 指定位置为对手子
                return 2
            else:
                if next:
                    return self._get_stone_color(Point(x, y), x_offset, y_offset, False)
                else:
                    return 0
        else:
            return 0
