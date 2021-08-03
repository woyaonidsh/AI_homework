from reinforcement.MCTS import MCTS
import time
from utils import RL
from RL_resnet import make_model
import os
import matplotlib.pyplot as plt
import torch
import config
from reinforcement.trainer_rl import Trainer

argus = config.parse_args()  # 加载参数


# child node的action我似乎是写错了，每一个node的child之内对应的每一个child node之中都应该有一个action

def main(MCTS_file=None, pretrained_model=None):
    # 确定训练设备
    argus.device = torch.cuda.is_available()
    device = torch.device("cuda:0" if argus.device else "cpu")  # 查看是否具有GPU
    print('*' * 80)
    print('The device: ', device)  # 输出当前设备名
    print('*' * 80)

    # 加载模型
    model = make_model(line_board=argus.Line_Points, hidden_dim=argus.rl_hidden,
                       input_layer=argus.input_layer, out_layer=argus.out_layer).to(device)

    # 构建优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=argus.lr, weight_decay=argus.wd)

    # 总网络
    if pretrained_model is not None:
        Net = torch.load(pretrained_model)
    else:
        Net = Trainer(model=model, optimizer=optimizer, device=device)

    # 保存游戏记录的队列
    stack = RL.random_stack()

    # 加载MCTS文件
    if MCTS_file:
        tree = RL.read_file(MCTS_file)
    else:

        tree = MCTS(board_size=argus.Line_Points, simulation_per_step=argus.simulation_per_step, neural_network=Net)

    # 记录训练情况
    record = []

    # 开始训练
    for epoch in range(argus.games):   # 总共进行的游戏数
        game_record, eval, steps = tree.game()   # 模拟的游戏步骤
        if len(game_record) % 2 == 1:
            print("game {} completed, black win, this game length is {}".format(epoch, len(game_record)))
        else:
            print("game {} completed, white win, this game length is {}".format(epoch, len(game_record)))
        print("The average eval:{}, the average steps:{}".format(eval, steps))
        train_data = RL.generate_training_data(game_record=game_record, board_size=RL.board_size)  # 产生训练数据
        for i in range(len(train_data)):
            stack.push(train_data[i])
        if epoch % 10 == 0 and epoch != 0:   # 每10局游戏训练网络
            my_loader = RL.generate_data_loader(stack)   # 创建数据集
            RL.write_file(my_loader, "../checkpoint/debug_loader.pkl")
            for _ in range(20):   # 重复训练5次
                record.extend(Net.train(my_loader, epoch))   # 训练网络
            print("train finished")
        if epoch % 200 == 0 and epoch != 0:
            torch.save(Net, "../checkpoint/rl_model_{}.pkl".format(epoch))    # 保存模型文件
            test_game_record, _, _ = tree.game(train=False)  # 测试模型
            RL.write_file(test_game_record, "../checkpoint/test_{}.pkl".format(epoch))  # 保存测试记录
            print("We finished a test game at {} game time".format(epoch))
        if epoch % 200 == 0 and epoch != 0:
            plt.figure()
            plt.plot(record)
            plt.title("cross entropy loss")
            plt.xlabel("step passed")
            plt.ylabel("Loss")
            plt.savefig("../checkpoint/loss_record_{}.svg".format(epoch))
            plt.close()


main()
print("here we are")


