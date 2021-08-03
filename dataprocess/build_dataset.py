import numpy as np
import os
import skimage
import skimage.transform
import skimage.io
from tqdm import tqdm
from copy import deepcopy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

tensor_path = 'D:\Homework\Datasets\AI_homework\\tensor\\'
image_path = 'D:\Homework\Datasets\AI_homework\image\\'
data_size = 10008  # 数据集大小
save_tensor = '../data/label/'  # 保存训练集,验证集,测试集对应的numpy数组
save_image = '../data/image/'


def process_data(path=tensor_path, image_path=image_path, data_size=data_size, save_tensor=save_tensor, save_image=save_image):
    files = os.listdir(path)
    dataset = []  # 总的数据集
    for i in range(1, len(files) + 1):
        file_path = path + 'train' + str(i) + '.npy'
        data = np.load(file_path)
        dataset.append(data)

    dataset = np.concatenate(dataset, axis=0)

    # 抽样数组
    indices = [_ for _ in range(data_size)]
    sample = [_ for _ in range(data_size)]
    indices = np.array(indices)
    sample = np.array(sample)
    np.random.shuffle(indices)
    np.random.shuffle(sample)

    print(indices)
    print(sample)

    train_split = []  # 训练集选中的数据
    val_split = []  # 验证集选中的数据
    test_split = []  # 测试集选中的数据
    for i in sample[:8000]:
        train_split.append(indices[i])

    for i in sample[8000:9000]:
        val_split.append(indices[i])

    for i in sample[9000:]:
        test_split.append(indices[i])

    train = []  # 训练集
    val = []  # 验证集
    test = []  # 测试集

    # 获取数据集
    for i in train_split:
        shuju = np.expand_dims(dataset[i], axis=0)
        train.append(shuju)
    for i in val_split:
        shuju = np.expand_dims(dataset[i], axis=0)
        val.append(shuju)
    for i in test_split:
        shuju = np.expand_dims(dataset[i], axis=0)
        test.append(shuju)

    train = np.concatenate(train, axis=0)
    val = np.concatenate(val, axis=0)
    test = np.concatenate(test, axis=0)

    # 保存数据集
    np.save(save_tensor + 'train.npy', train)
    np.save(save_tensor + 'val.npy', val)
    np.save(save_tensor + 'test.npy', test)

    # 保存图片
    jishu = 0
    for i in tqdm(train_split, desc='The training data: '):
        image_file = os.path.join(image_path, str(i) + '.png')
        image = skimage.io.imread(image_file)
        new_image = deepcopy(image)
        filename = os.path.join(save_image, 'train/' + str(jishu) + '.png')
        skimage.io.imsave(filename, new_image)  # 保存处理好的图片
        jishu += 1
    jishu = 0
    for i in tqdm(val_split, desc='The val data: '):
        image_file = os.path.join(image_path, str(i) + '.png')
        image = skimage.io.imread(image_file)
        new_image = deepcopy(image)
        filename = os.path.join(save_image, 'val/' + str(jishu) + '.png')
        skimage.io.imsave(filename, new_image)
        jishu += 1
    jishu = 0
    for i in tqdm(test_split, desc='The test data: '):
        image_file = os.path.join(image_path, str(i) + '.png')
        image = skimage.io.imread(image_file)
        new_image = deepcopy(image)
        filename = os.path.join(save_image, 'test/' + str(jishu) + '.png')
        skimage.io.imsave(filename, new_image)
        jishu += 1

    print('training data size: ', len(train))
    print('val data size: ', len(val))
    print('test data size: ', len(test))
