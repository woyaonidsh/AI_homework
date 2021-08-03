import sys
import random
import pygame
from pygame.locals import *
import pygame.gfxdraw
import numpy as np
from copy import deepcopy

from checkerboard import Checkerboard, BLACK_CHESSMAN, WHITE_CHESSMAN, Point
from Draw_board import draw_board
import config
from AI_computer import AI

argus = config.parse_args()

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


def print_text(screen, font, x, y, text, fcolor=(255, 255, 255)):
    imgText = font.render(text, True, fcolor)
    screen.blit(imgText, (x, y))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('五子棋')

    font1 = pygame.font.SysFont('SimHei', 32)  # 加载字体模块
    font2 = pygame.font.SysFont('SimHei', 72)
    fwidth, fheight = font2.size('黑方获胜')

    checkerboard = Checkerboard(Line_Points)  # 初始化棋盘
    cur_runner = BLACK_CHESSMAN  # 黑棋先走
    winner = None  # 无胜者
    computer = AI(Line_Points, WHITE_CHESSMAN)  # AI

    black_win_count = 0
    white_win_count = 0

    #    image_name = 9978
    #    train_data = []

    draw = draw_board()  # 画图的类

    # 画图
    draw.Draw(screen=screen, checkerboard=checkerboard.checkerboard, font1=font1, cur_runner=cur_runner,
              black_win_count=black_win_count, white_win_count=white_win_count)

    while True:
        for event in pygame.event.get():  # 监听用户事件
            if event.type == QUIT:  # 用户退出游戏
                #                numpy_data = np.array(train_data)
                #                np.save(argus.save_tensor, numpy_data)    # 保存图片的矩阵
                #                print(len(numpy_data))
                sys.exit()
            elif event.type == KEYDOWN:  # 用户按下键盘
                if event.key == K_RETURN:
                    if winner is not None:
                        winner = None
                        cur_runner = BLACK_CHESSMAN  # 当前下棋者
                        checkerboard = Checkerboard(Line_Points)
                        computer = AI(Line_Points, WHITE_CHESSMAN)
            elif event.type == MOUSEBUTTONDOWN:  # 用户使用鼠标
                if winner is None:
                    pressed_array = pygame.mouse.get_pressed()  # 鼠标是否被按下
                    if pressed_array[0]:
                        mouse_pos = pygame.mouse.get_pos()  # 获取鼠标位置
                        click_point = _get_clickpoint(mouse_pos)  # 获取落子位置
                        if click_point is not None:  # 判断落子是否有效
                            if checkerboard.can_drop(click_point):  # 判断是否可落子
                                winner = checkerboard.drop(cur_runner, click_point)  # 落子判断是否胜出

                                # 画图
                                draw.Draw(screen=screen, checkerboard=checkerboard.checkerboard, font1=font1,
                                          cur_runner=cur_runner, black_win_count=black_win_count,
                                          white_win_count=white_win_count)
                                # print(checkerboard.checkerboard) train_data.append(deepcopy(
                                # checkerboard.checkerboard))   # 添加照片的矩阵 pygame.image.save(screen, argus.save_image
                                # + str(image_name) + '.png') image_name += 1

                                if winner is None:
                                    cur_runner = _get_next(cur_runner)  # 下一人落子
                                    computer.get_opponent_drop(click_point)
                                    AI_point = computer.AI_drop()  # AI落子位置
                                    winner = checkerboard.drop(cur_runner, AI_point)

                                    # 画图
                                    draw.Draw(screen=screen, checkerboard=checkerboard.checkerboard, font1=font1,
                                              cur_runner=cur_runner, black_win_count=black_win_count,
                                              white_win_count=white_win_count)
                                    # print(checkerboard.checkerboard) train_data.append(deepcopy(
                                    # checkerboard.checkerboard)) pygame.image.save(screen, argus.save_image + str(
                                    # image_name) + '.png') image_name += 1

                                    if winner is not None:
                                        white_win_count += 1  # 机器赢数
                                    cur_runner = _get_next(cur_runner)
                                else:
                                    black_win_count += 1  # 人赢数
                        else:
                            print('超出棋盘区域')

        if winner:
            print_text(screen, font2, (SCREEN_WIDTH - fwidth) // 2, (SCREEN_HEIGHT - fheight) // 2, winner.Name + '获胜',
                       RED_COLOR)

        # 更细屏幕显示
        pygame.display.flip()


# 返回下一个该落子的人
def _get_next(cur_runner):
    if cur_runner == BLACK_CHESSMAN:
        return WHITE_CHESSMAN
    else:
        return BLACK_CHESSMAN


# 根据鼠标点击位置，返回游戏区坐标
def _get_clickpoint(click_pos):
    pos_x = click_pos[0] - Start_X
    pos_y = click_pos[1] - Start_Y
    if pos_x < -Inside_Width or pos_y < -Inside_Width:
        return None
    x = pos_x // SIZE
    y = pos_y // SIZE
    if pos_x % SIZE > Stone_Radius:
        x += 1
    if pos_y % SIZE > Stone_Radius:
        y += 1
    if x >= Line_Points or y >= Line_Points:
        return None

    return Point(x, y)


if __name__ == '__main__':
    main()
