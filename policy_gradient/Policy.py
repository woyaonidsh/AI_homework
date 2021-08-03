import sys
import pygame
from pygame.locals import *
import pygame.gfxdraw
import torch
from copy import deepcopy
import time

from checkerboard import Checkerboard, BLACK_CHESSMAN, WHITE_CHESSMAN, Point
from Draw_board import draw_board
import config
from PG_resnet import make_model
from utils import PG
from trainer_pg import Trainer
from reward import get_score

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
    # 确定训练设备
    argus.device = torch.cuda.is_available()
    device = torch.device("cuda:0" if argus.device else "cpu")  # 查看是否具有GPU
    print('*' * 80)
    print('The device: ', device)  # 输出当前设备名
    print('*' * 80)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('五子棋')

    font1 = pygame.font.SysFont('SimHei', 32)  # 加载字体模块
    font2 = pygame.font.SysFont('SimHei', 72)
    fwidth, fheight = font2.size('黑方获胜')

    checkerboard = Checkerboard(Line_Points)  # 初始化棋盘
    cur_runner = BLACK_CHESSMAN  # 黑棋先走
    winner = None  # 无胜者

    black_win_count = 0  # 黑子赢的局数
    white_win_count = 0  # 白子赢的局数
    total_count = 0  # 总的对局数

    draw = draw_board()  # 画图的类

    # 画图
    draw.Draw(screen=screen, checkerboard=checkerboard.checkerboard, font1=font1, cur_runner=cur_runner,
              black_win_count=black_win_count, white_win_count=white_win_count)

    # 加载模型
    model = make_model(line_board=argus.Line_Points, hidden_dim=argus.rl_hidden,
                       input_layer=argus.input_layer, out_layer=argus.out_layer)

    # 加载优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=argus.lr, weight_decay=argus.wd)

    # 加载训练器
    trainer = Trainer(model=model, optimizer=optimizer, batchsize=argus.batch_size, device=device)

    # 计算得分
    reward = get_score(board_line=argus.Line_Points, my_value=2, opponent_value=1)

    # 初始保存数据的队列
    stack = PG.dataset()

    # 当前得分
    all_score = 0

    # 当前局的游戏记录
    game_state = []
    game_action = []
    game_reward = []
    game_winner = []

    while True:
        for event in pygame.event.get():  # 监听用户事件
            if event.type == QUIT:  # 用户退出游戏
                sys.exit()
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

                                if winner is None:
                                    game_state.append(deepcopy(checkerboard.checkerboard))  # 保存当前棋盘棋局

                                    cur_runner = _get_next(cur_runner)  # 下一人落子
                                    AI_point = trainer.eval(checkerboard.checkerboard)

                                    action, p_action = PG.select_action(AI_point, checkerboard.checkerboard)  # AI落子位置
                                    game_action.append(deepcopy(p_action))  # 保存AI的落子

                                    AI_point = Point(action[0], action[1])

                                    winner = checkerboard.drop(cur_runner, AI_point)

                                    score = reward.all_score(checkerboard.checkerboard)
                                    game_reward.append(score)  # 保存当前棋局的得分

                                    # 画图
                                    draw.Draw(screen=screen, checkerboard=checkerboard.checkerboard, font1=font1,
                                              cur_runner=cur_runner, black_win_count=black_win_count,
                                              white_win_count=white_win_count)

                                    if winner is not None:
                                        white_win_count += 1  # 机器赢数
                                        total_count += 1

                                        for length in range(len(game_reward)):
                                            game_winner.append(1)

                                        # 计算当前reward
                                        all_reward = PG.compute_reward(game_reward, game_winner)

                                        # 装载数据
                                        item = {'state': game_state, 'distribution': game_action,
                                                'reward': all_reward, 'value': game_winner}
                                        stack.push(item)

                                        all_score += sum(all_reward) / len(all_reward)   # 计算得分

                                        # 开始训练
                                        if total_count % 10 == 0 and total_count != 0:
                                            # 构建数据集
                                            dataset = PG.data_loader(stack, batch_size=argus.batch_size)
                                            # 开始训练模型
                                            loss = trainer.train(data_loader=dataset)
                                            print('The average reward is :', all_score / 10)
                                            print('The now reward is :', sum(all_reward) / len(all_reward))
                                            all_score = 0
                                            stack.renew()

                                        checkerboard = Checkerboard(Line_Points)  # 初始化棋盘

                                        # 清空当前对局记录
                                        game_state = []
                                        game_action = []
                                        game_reward = []
                                        game_winner = []

                                    cur_runner = _get_next(cur_runner)
                                else:
                                    black_win_count += 1  # 人赢数
                                    total_count += 1
                                    for length in range(len(game_reward)):
                                        game_winner.append(-1)

                                    # 计算当前reward
                                    all_reward = PG.compute_reward(game_reward, game_winner)

                                    # 装载数据
                                    item = {'state': game_state, 'distribution': game_action,
                                            'reward': all_reward, 'value': game_winner}
                                    stack.push(item)

                                    all_score += sum(all_reward) / len(all_reward)

                                    # 开始训练
                                    if total_count % 10 == 0 and total_count != 0:
                                        # 构建数据集
                                        dataset = PG.data_loader(stack, batch_size=argus.batch_size)
                                        # 开始训练模型
                                        loss = trainer.train(data_loader=dataset)
                                        print('The average reward is :', all_score / 10)
                                        print('The reward is :', sum(all_reward) / len(all_reward))
                                        all_score = 0
                                        stack.renew()

                                    checkerboard = Checkerboard(Line_Points)  # 初始化棋盘

                                    # 清空当前对局记录
                                    game_state = []
                                    game_action = []
                                    game_reward = []
                                    game_winner = []
                        else:
                            print('超出棋盘区域')

        if winner:
            print_text(screen, font2, (SCREEN_WIDTH - fwidth) // 2, (SCREEN_HEIGHT - fheight) // 2, winner.Name + '获胜',
                       RED_COLOR)
            pygame.display.flip()

            winner = None
            checkerboard = Checkerboard(Line_Points)  # 初始化棋盘
            cur_runner = BLACK_CHESSMAN  # 黑棋先走

            time.sleep(1)   # 让画面暂停一段时间

            # 画图
            draw.Draw(screen=screen, checkerboard=checkerboard.checkerboard, font1=font1, cur_runner=cur_runner,
                      black_win_count=black_win_count, white_win_count=white_win_count)

        # 更新屏幕显示
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
