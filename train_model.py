import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import json

from model import resnet
import config
from dataprocess import load_data
from utils import metric


argus = config.parse_args()


class Trainer:
    def __init__(self, model, criterion, optimizer, batchsize, device):
        self.epoch = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batchsize = batchsize
        self.device = device

    def train(self, dataset):
        images, labels, datasize = dataset  # 加载训练集
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = [_ for _ in range(datasize)]
        indices = np.array(indices)
        np.random.shuffle(indices)    # 打乱顺序
        for i in tqdm(range(datasize), desc='Training epoch ' + str(self.epoch + 1) + ''):
            sample = indices[i]
            label = labels[sample].to(self.device)  # 获取h 一笔数据
            image = torch.tensor(images[sample], dtype=torch.float).unsqueeze(0).to(self.device)
            output = self.model(image)  # 模型的输入
            loss = self.criterion(output, label)  # 获得loss
            total_loss += loss.item()
            loss.backward()
            if i % self.batchsize == 0 and i > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                print('Loss is: %.4f' % (total_loss / i))
        self.epoch += 1
        return total_loss / datasize

    @torch.no_grad()
    def val(self, dataset):
        with torch.no_grad():
            images, labels, datasize = dataset  # 加载验证集
            total_loss = 0.0
            right = 0  # 预测对的数
            for i in tqdm(range(datasize), desc='Training epoch ' + str(self.epoch) + ''):
                label = labels[i].to(self.device)  # 获取一笔数据
                image = torch.tensor(images[i], dtype=torch.float).unsqueeze(0).to(self.device)
                output = self.model(image)  # 模型的输入
                right += metric.equal(prediction=output, label=label)
                loss = self.criterion(output, label)  # 获得loss
                total_loss += loss.item()
            return {'loss': total_loss / datasize, 'Accuracy': right / datasize}

    @torch.no_grad()
    def test(self, dataset):
        with torch.no_grad():
            images, labels, datasize = dataset  # 加载验证集
            right = 0  # 预测对的数
            for i in tqdm(range(datasize), desc='Training epoch ' + str(self.epoch) + ''):
                label = labels[i].to(self.device)  # 获取一笔数据
                image = torch.tensor(images[i], dtype=torch.float).unsqueeze(0).to(self.device)
                output = self.model(image)  # 模型的输入
                right += metric.equal(prediction=output, label=label)
            return {'Accuracy': right / datasize}


def main():
    argus.device = torch.cuda.is_available()
    device = torch.device("cuda:0" if argus.device else "cpu")  # 查看是否具有GPU
    print('*' * 80)
    print('The device: ', device)  # 输出当前设备名
    print('*' * 80)
    # 加载数据
    Data = load_data.Load_data(image_path=argus.image_path, tensor_path=argus.label_path)

    # 加载模型
    model = resnet.resnet50().to(device)
    # model = make_model(hidden_dim=argus.hidden_dim, d_ff=argus.d_ff, head=argus.head, layer=argus.layer,
    # resnet_path=argus.resnet_path, pretrained=argus.pretrained, image_dim=argus.image_dim).to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    print('The loss function: CrossEntropyLoss')

    # 优化器
    if argus.optim == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=argus.lr, weight_decay=argus.wd)
        print('The optimizer : Adam')
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), momentum=0.9, lr=argus.lr, weight_decay=argus.wd)
        print('The optimizer: SGD')

    # 构建训练器
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer,
                      batchsize=argus.batch_size, device=device)

    log_file = open(argus.save_log, 'a', encoding='utf-8')   # 记录训练信息文件

    # 开始训练
    for i in range(argus.epochs):
        print('Start training model: ', '\n')
        train_loss = trainer.train(Data[0])

        print('Start validating data: ', '\n')
        val = trainer.val(Data[1])

        print('Start testing data: ', '\n')
        test = trainer.test(Data[2])

        print('The training loss: ', train_loss)
        print('The val: ', val)
        print('The test: ', test)

        # 保存训练信息
        information = {
            'epoch': i + 1,
            'train_loss': train_loss,
            'val': val,
            'test': test
        }
        information = json.dumps(information)
        log_file.write(information)
        log_file.write('\n')

        # 保存模型信息
        torch.save(model.state_dict(), argus.save_model)

    log_file.close()


if __name__ == "__main__":
    main()
