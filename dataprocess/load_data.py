import skimage
import skimage.transform
import skimage.io
import torch
import os
import numpy as np
from copy import copy
from tqdm import tqdm


class Load_data:
    def __init__(self, image_path, tensor_path):
        self.image_path = image_path
        self.tensor_path = tensor_path
        self.data_size = []
        self.image, self.label = self.load()

    def __getitem__(self, index):  # 可以让对象具有迭代功能
        image = copy(self.image[index])
        label = copy(self.label[index])
        datasize = copy(self.data_size[index])
        return image, label, datasize  # 返回最终数据

    def load(self):
        print('start loading data: ', '\n')
        dataset = ['train', 'val', 'test']
        tensor_data = []  # 图片的矩阵
        for data in tqdm(dataset, desc='loading label data: '):
            tensor_path = os.path.join(self.tensor_path, data + '.npy')
            train = np.load(tensor_path)
            train = torch.tensor(train, dtype=torch.int64)  # 转换为torch张量
            tensor_data.append(train)
            self.data_size.append(len(train))

        image_data = []
        for data in dataset:
            image_path = os.path.join(self.image_path, data + '/')
            image_file = os.listdir(image_path)

            # get_key是sotred函数用来比较的元素，该处用lambda表达式替代函数。
            key = lambda ii: int(ii.split('.')[0])
            images = sorted(image_file, key=key)

            train_image = []
            # 读取图片
            for i in tqdm(images, desc='loading image data: '):
                file = os.path.join(image_path, i)
                image = skimage.io.imread(file)
                picture = image.transpose(2, 0, 1)
                train_image.append(picture)
            image_data.append(train_image)

        return image_data, tensor_data


"""
model = Load_data(image_path='../data/image/', tensor_path='../data/label/')

images, labels, datasize = model[0]

print(labels[0])
print(labels[1])
print(labels[2])
print([labels[1]])
"""