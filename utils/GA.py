import torch
from torch.distributions import Categorical
import torch.utils.data as torch_data

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


class dataset:
    def __init__(self, length=3000):
        self.state = []  # 当前棋盘状态
        self.distrib = []  # 当前模型输出分布
        self.reward = []  # 每一轮游戏的总奖励值
        self.winner = []  # 这一局的获胜者
        self.length = length

    def isEmpty(self):
        return len(self.state) == 0

    def push(self, item):
        self.state.extend(item['state'])
        self.distrib.extend(item['distribution'])
        self.reward.extend(item['reward'])
        self.winner.extend(item['value'])
        if len(self.state) >= self.length:
            self.state = self.state[1:]
            self.distrib = self.distrib[1:]
            self.winner = self.winner[1:]

    def get(self):  # 获取当前保存的数据
        return self.state, self.distrib, self.reward, self.winner

    def renew(self):  # 清空对局数据
        self.state = []
        self.distrib = []
        self.reward = []
        self.winner = []


# 抽样
def select_action(probs, check_board):
    position = get_mask(check_board)
    probs = torch.mul(probs, position)
    probs = torch.softmax(probs, dim=-1)
    m = Categorical(probs)  # 生成分布
    action = m.sample()  # 从分布中采样
    X = action.item() % 15
    Y = action.item() // 15
    target = get_target(action.item())
    return [X, Y], target  # 返回一个元素值


# 构建数据集
def data_loader(stack, batch_size):
    state, distrib, reward, winner = stack.get()
    tensor_x = torch.stack(tuple([torch.tensor(s).unsqueeze(0) for s in state]))  # 棋局状态
    tensor_y1 = torch.stack(tuple([y1 for y1 in distrib]))  # 棋局得分分布
    tensor_y2 = torch.stack(tuple([torch.tensor([y2]) for y2 in reward]))  # 获取奖励
    tensor_y3 = torch.stack(tuple([torch.tensor([y3]) for y3 in winner]))  # 棋局获胜者
    dataset = torch_data.TensorDataset(tensor_x, tensor_y1, tensor_y2, tensor_y3)
    my_loader = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return my_loader


# 计算一局比赛的reward
def compute_reward(game_reward, game_winner):
    reward = sum(game_reward)
    if game_winner[0] == 1:
        score = len(game_winner) / 2
        reward += score
    else:
        score = len(game_winner)
        reward -= score
    all_reward = [reward] * len(game_reward)
    return all_reward


# 落有点的位置进行遮盖
def get_mask(state):
    target = []
    for i in state:
        for j in i:
            if j == 0:
                target.append(1)
            else:
                target.append(0)
    target = torch.tensor([target])
    return target


# 生成标签数据
def get_target(position):
    target = [0] * 225
    target = torch.tensor([target])
    target[0][position] = 1
    return target