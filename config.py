import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='AI homework')

    # TODO -----------保存文件---------------------
    parser.add_argument('--save_image', type=str, default='data/image2/')
    parser.add_argument('--save_tensor', type=str, default='data/tensor/train160.npy')

    # TODO -----------数据文件---------------------
    parser.add_argument('--image_path', type=str, default='data/image/')
    parser.add_argument('--label_path', type=str, default='data/label/')

    # TODO ----------棋盘-------------------------
    parser.add_argument('--SIZE', type=int, default=30)  # 棋盘每个点时间的间隔
    parser.add_argument('--Line_Points', type=int, default=15)  # 棋盘每行/每列点数
    parser.add_argument('--Outer_Width', type=int, default=20)  # 棋盘外宽度
    parser.add_argument('--Border_Width', type=int, default=4)  # 边框宽度
    parser.add_argument('--Inside_Width', type=int, default=4)  # 边框跟实际的棋盘之间的间隔

    # TODO ----------网络--------------------------
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--resnet_path', type=str, default='pretrained/resnet50')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--image_dim', type=int, default=1000)

    # TODO ---------训练网络-------------------------
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0)

    # TODO ---------模型文件--------------------------
    parser.add_argument('--save_model', type=str, default='checkpoint/model.pth')
    parser.add_argument('--save_log', type=str, default='checkpoint/log.json')

    # TODO --------强化学习文件参数---------------------
    parser.add_argument('--rl_model', type=str, default='checkpoint/rl_model.pth')
    parser.add_argument('--MCTS_file', type=str, default='checkpoint/MCTS_model.pth')

    # TODO --------强化学习模型参数---------------------
    parser.add_argument('--rl_hidden', type=int, default=100)
    parser.add_argument('--input_layer', type=int, default=1)
    parser.add_argument('--out_layer', type=str, default=1)
    parser.add_argument('--simulation_per_step', type=int, default=200)  # 模拟搜索步数
    parser.add_argument('--games', type=int, default=1000)  # 总共自对弈游戏数

    args = parser.parse_args()
    return args
