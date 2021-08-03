import torch
import torch.nn as nn
from torch.autograd import Variable


class Trainer:
    def __init__(self, model, optimizer, device):
        super(Trainer, self).__init__()
        self.model = model   # 模型
        self.optimizer = optimizer   # 优化器
        self.device = device
        self.mseloss = nn.MSELoss()

    def train(self, data_loader, game_time):  # 只有在训练的时候才会用的到的train
        self.model.train()
        loss_record = []
        for batch_idx, (state, distrib, winner) in enumerate(data_loader):
            tmp = []
            state, distrib, winner = Variable(state).float().to(self.device), \
                                     Variable(distrib).float().to(self.device),  \
                                     Variable(winner).float().to(self.device)
            self.optimizer.zero_grad()
            prob, value = self.model(state)
            output = torch.log_softmax(prob, dim=1)   # 落子概率

            cross_entropy = - torch.mean(torch.sum(distrib * output, 1))
            mse = self.mseloss(value, winner)
            loss = mse + cross_entropy
            loss.backward()

            self.optimizer.step()
            tmp.append(cross_entropy.data)
            if batch_idx % 10 == 0:
                print("We have played {} games, and batch {}, the cross entropy loss is {}, the mse loss is {}".format(
                    game_time, batch_idx, cross_entropy.data, mse.data))
                loss_record.append(sum(tmp) / len(tmp))
        return loss_record

    def eval(self, state):  # 会用到很多次的eval
        self.model.eval()
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            prob, value = self.model(state)
        return torch.softmax(prob, dim=1), value