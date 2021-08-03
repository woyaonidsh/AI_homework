import torch
import torch.nn as nn
from torch.autograd import Variable


class Trainer:
    def __init__(self, model, optimizer, batchsize, device):
        self.model = model
        self.optimizer = optimizer
        self.batchsize = batchsize
        self.device = device

    def train(self, data_loader):
        self.model.to(self.device)
        loss_record = []
        total_loss = 0
        self.optimizer.zero_grad()
        for batch_idx, (state, distrib, reward, winner) in enumerate(data_loader):
            state, distrib, reward, winner = Variable(state).float().to(self.device), \
                                             Variable(distrib).float().squeeze(0).to(self.device), \
                                             Variable(reward).float().to(self.device), \
                                             Variable(winner).float().to(self.device)
            pro = self.model(state).unsqueeze(1)
            output = torch.log_softmax(pro, dim=-1)
            cross_entropy = torch.sum(output * distrib, dim=-1)
            loss = torch.mean(- cross_entropy * winner, dim=0)    # 策略梯度
            total_loss += loss.item()
            loss.backward()
            if batch_idx % 2 == 0 and batch_idx != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_record.append(total_loss / batch_idx)
        if len(loss_record) != 0:
            print('The total loss : ', sum(loss_record) / len(loss_record))
        return loss_record

    def eval(self, state):
        self.model.to('cpu')
        state = torch.tensor(state).float().unsqueeze(0).unsqueeze(0)   # 当前棋盘状态
        with torch.no_grad():
            pro = self.model(state)
        return torch.softmax(pro, dim=-1)  # 得到当前的概率输出
