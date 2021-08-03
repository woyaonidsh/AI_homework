import pygame
import pygame.gfxdraw
from collections import namedtuple

SIZE = 30  # 棋盘每个点时间的间隔
Line_Points = 15  # 棋盘每行/每列点数
Outer_Width = 20  # 棋盘外宽度
Border_Width = 4  # 边框宽度
Inside_Width = 4  # 边框跟实际的棋盘之间的间隔
Border_Length = SIZE * (Line_Points - 1) + Inside_Width * 2 + Border_Width  # 边框线的长度
Start_X = Start_Y = Outer_Width + int(Border_Width / 2) + Inside_Width  # 网格线起点（左上角）坐标
SCREEN_HEIGHT = SIZE * (Line_Points - 1) + Outer_Width * 2 + Border_Width + Inside_Width * 2  # 游戏屏幕的高
SCREEN_WIDTH = SCREEN_HEIGHT + 200  # 游戏屏幕的宽

Stone_Radius = SIZE // 2 - 3  # 棋子半径
Stone_Radius2 = SIZE // 2 + 3
Checkerboard_Color = (0xE3, 0x92, 0x65)  # 棋盘颜色
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (200, 30, 30)
BLUE_COLOR = (30, 30, 200)

RIGHT_INFO_POS_X = SCREEN_HEIGHT + Stone_Radius2 * 2 + 10

Chessman = namedtuple('Chessman', 'Name Value Color')
Point = namedtuple('Point', 'X Y')

BLACK_CHESSMAN = Chessman('黑子', 1, (45, 45, 45))
WHITE_CHESSMAN = Chessman('白子', 2, (219, 219, 219))


class draw_board:
    def __init__(self):
        self.draw = '画棋盘'

    def print_text(self, screen, font, x, y, text, fcolor=(255, 255, 255)):
        imgText = font.render(text, True, fcolor)
        screen.blit(imgText, (x, y))

    # 画棋盘
    def draw_checkerboard(self, screen):
        # 填充棋盘背景色
        screen.fill(Checkerboard_Color)
        # 画棋盘网格线外的边框
        pygame.draw.rect(screen, BLACK_COLOR, (Outer_Width, Outer_Width, Border_Length, Border_Length), Border_Width)
        # 画网格线
        for i in range(Line_Points):
            pygame.draw.line(screen, BLACK_COLOR,
                             (Start_Y, Start_Y + SIZE * i),
                             (Start_Y + SIZE * (Line_Points - 1), Start_Y + SIZE * i),
                             1)
        for j in range(Line_Points):
            pygame.draw.line(screen, BLACK_COLOR,
                             (Start_X + SIZE * j, Start_X),
                             (Start_X + SIZE * j, Start_X + SIZE * (Line_Points - 1)),
                             1)
        # 画星位和天元
        for i in (3, 9, 15):
            for j in (3, 9, 15):
                if i == j == 9:
                    radius = 5
                else:
                    radius = 3
                # pygame.draw.circle(screen, BLACK, (Start_X + SIZE * i, Start_Y + SIZE * j), radius)
                pygame.gfxdraw.aacircle(screen, Start_X + SIZE * i, Start_Y + SIZE * j, radius, BLACK_COLOR)
                pygame.gfxdraw.filled_circle(screen, Start_X + SIZE * i, Start_Y + SIZE * j, radius, BLACK_COLOR)

    # 画棋子
    def draw_chessman(self, screen, point, stone_color):
        # pygame.draw.circle(screen, stone_color, (Start_X + SIZE * point.X, Start_Y + SIZE * point.Y), Stone_Radius)
        pygame.gfxdraw.aacircle(screen, Start_X + SIZE * point.X, Start_Y + SIZE * point.Y, Stone_Radius, stone_color)
        pygame.gfxdraw.filled_circle(screen, Start_X + SIZE * point.X, Start_Y + SIZE * point.Y, Stone_Radius,
                                     stone_color)

    # 画左侧信息显示
    def draw_left_info(self, screen, font, cur_runner, black_win_count, white_win_count):
        self.draw_chessman_pos(screen, (SCREEN_HEIGHT + Stone_Radius2, Start_X + Stone_Radius2), BLACK_CHESSMAN.Color)
        self.draw_chessman_pos(screen, (SCREEN_HEIGHT + Stone_Radius2, Start_X + Stone_Radius2 * 4),
                               WHITE_CHESSMAN.Color)

        self.print_text(screen, font, RIGHT_INFO_POS_X, Start_X + 3, '玩家', BLUE_COLOR)
        self.print_text(screen, font, RIGHT_INFO_POS_X, Start_X + Stone_Radius2 * 3 + 3, '电脑', BLUE_COLOR)

        self.print_text(screen, font, SCREEN_HEIGHT, SCREEN_HEIGHT - Stone_Radius2 * 8, '战况：', BLUE_COLOR)
        self.draw_chessman_pos(screen, (SCREEN_HEIGHT + Stone_Radius2, SCREEN_HEIGHT - int(Stone_Radius2 * 4.5)),
                           BLACK_CHESSMAN.Color)
        self.draw_chessman_pos(screen, (SCREEN_HEIGHT + Stone_Radius2, SCREEN_HEIGHT - Stone_Radius2 * 2),
                           WHITE_CHESSMAN.Color)
        self.print_text(screen, font, RIGHT_INFO_POS_X, SCREEN_HEIGHT - int(Stone_Radius2 * 5.5) + 3,
                        f'{black_win_count} 胜', BLUE_COLOR)
        self.print_text(screen, font, RIGHT_INFO_POS_X, SCREEN_HEIGHT - Stone_Radius2 * 3 + 3,
                        f'{white_win_count} 胜', BLUE_COLOR)

    def draw_chessman_pos(self, screen, pos, stone_color):
        pygame.gfxdraw.aacircle(screen, pos[0], pos[1], Stone_Radius2, stone_color)
        pygame.gfxdraw.filled_circle(screen, pos[0], pos[1], Stone_Radius2, stone_color)

    # 画图函数
    def Draw(self, screen, checkerboard, font1, cur_runner, black_win_count, white_win_count):
        # 画棋盘
        self.draw_checkerboard(screen)

        # 画棋盘上棋子
        for i, row in enumerate(checkerboard):
            for j, cell in enumerate(row):
                if cell == BLACK_CHESSMAN.Value:
                    self.draw_chessman(screen, Point(j, i), BLACK_CHESSMAN.Color)
                elif cell == WHITE_CHESSMAN.Value:
                    self.draw_chessman(screen, Point(j, i), WHITE_CHESSMAN.Color)

        # 画左侧信息显示
        self.draw_left_info(screen, font1, cur_runner, black_win_count, white_win_count)