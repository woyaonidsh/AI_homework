import numpy as np
import torch
import random


class Genetic:
    def __init__(self, group, sigma, crate, mute, generation, Fitness):
        self.group = group   # 种群总数
        self.sigma = sigma   # 适应度阈值
        self.crate = crate  # 组合概率
        self.mute = mute    # 变异概率
        self.generation = generation  # 总共繁衍多少代
        self.fit = Fitness   # 适应度函数

    # 产生初代
    def init_generation(self):
        generation = torch.randn(2 * self.group, 225)
        generation = torch.softmax(generation, dim=-1)
        return generation

    # 计算适应度
    def evaluation(self, group):
        fit = []
        for i in group:   # 计算适应度
            result = self.fit(i)
            fit.append(result)
        return fit

    # 组合
    def combination(self, one, two):
        new_one = []
        new_two = []
        all = 225   # 总长度
        for length in range(all):
            r = random.randint(0, 100)
            if r % 5 != 0:
                new_one.append(two[length])
                new_two.append(one[length])
            else:
                new_one.append(one[length])
                new_two.append(two[length])
        new_one = torch.tensor(new_one)
        new_two = torch.tensor(new_two)
        return new_one, new_two

    # 变异
    def variation(self, one):
        r = random.randint(0, 100)
        substitution = [_ for _ in range(225)]
        substitution = np.array(substitution)
        np.random.shuffle(substitution)
        if r % 10 == 0:
            new_one = []
            for i in substitution:
                new_one.append(one[i])
            new_one = torch.tensor(new_one)
        else:
            new_one = one
        return new_one

    # 遗传操作
    def Genetic(self, group):
        new_group = []  # 新的种群
        indices = [_ for _ in range(self.group)]
        indices = np.array(indices)
        np.random.shuffle(indices)    # 打乱顺序

        # 进行组合操作
        for i in range(0, len(indices), 2):
            front = group[i]
            behind = group[i+1]
            new_one, new_two = self.combination(front, behind)
            new_group.append(new_one)  # 新组合的个体
            new_group.append(new_two)  # 新组合的个体

        f_group = []
        # 进行变异操作
        for j in range(len(new_group)):
            new = self.variation(new_group[j])
            f_group.append(new)
        return f_group

    # 选择
    def selection(self, group):
        fit = self.evaluation(group)
        new = []
        for i in range(len(fit)):
            new.append((fit[i], group[i]))

        # 根据适应度排序
        new.sort(key=lambda x: (x[0]), reverse=True)
        new = new[0: self.group]  # 保留前n个个体成为新的一代

        new_generation = []
        for j in range(self.group):
            new_generation.append(new[j][1])
        return new_generation

    # 判断是否已经达到阈值要求
    def check(self, fit):
        if fit > self.sigma:
            result = True
        else:
            result = False
        return result

    # 总函数
    def create(self):
        group = self.init_generation()
        new_group = self.selection(group)  # 第一代

        for i in range(self.generation):   # 更新n代
            new_group = self.Genetic(new_group)   # 遗传操作
            new_group = self.selection(new_group)   # 选择新一代
            result = self.check(new_group[0])
            if result:
                break
        return new_group[0]   # 返回子代中适应度最高的个体作为模型的落子


