import matplotlib.pyplot as plt


def draw_loss(x, y, z):
    plt.figure(figsize=(15, 10))  # 设置画布的尺寸
    plt.title('LOSS——Epochs', fontsize=20)  # 标题，并设定字号大小
    plt.xlabel('Epochs', fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel('LOSS', fontsize=14)  # 设置y轴，并设定字号大小
    plt.plot(x, y, color='red', marker='*', linewidth=2, label='training loss')
    plt.plot(x, z, marker='o', linewidth=1.5, linestyle=':', label='val loss')
    plt.legend(loc=1, frameon=False, fontsize='xx-large')  # 图例展示位置，数字代表第几象限
    plt.show()


def draw_accuracy(x, y, z):
    plt.figure(figsize=(15, 10))  # 设置画布的尺寸
    plt.title('Accuracy——Epochs', fontsize=20)  # 标题，并设定字号大小
    plt.xlabel('Epochs', fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel('Accuracy', fontsize=14)  # 设置y轴，并设定字号大小
    plt.plot(x, y, marker='*', linestyle=':', linewidth=2, label='val Accuracy')
    plt.plot(x, z, color='red', linestyle=':', marker='o', linewidth=2, label='test Accuracy')
    plt.legend(loc=2, frameon=False, fontsize='xx-large')  # 图例展示位置，数字代表第几象限
    plt.show()


# 数据均来源于模型的记录文件 log.txt
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
train_loss = [16.58, 0.56, 0.33, 0.18, 0.09, 0.036, 0.017, 0.009, 0.005, 0.005, 0.0035, 0.0029]
val_loss = [0.75, 0.38, 0.26, 0.13, 0.04, 0.022, 0.011, 0.0054, 0.0039, 0.00308, 0.0015, 0.0014]

val_accuracy = [0.0, 0.0, 0.012, 0.22, 0.433, 0.659, 0.796, 0.896, 0.935, 0.921, 0.983, 0.97]

test_accuracy = [0.0, 0.0, 0.013, 0.202, 0.446, 0.6517, 0.7886, 0.9166, 0.9335, 0.9146, 0.9821, 0.9702]

# 画出模型的loss曲线
# draw_loss(x=epoch, y=train_loss, z=val_loss)

# 画出模型的accuracy曲线
# draw_accuracy(x=epoch, y=val_accuracy, z=test_accuracy)


reward = [-20, -19.5, -18.1, -19, -20.8, -16.4, -18.2, -20.8, -20.5, -19.2, -18.9, -20.4, -19.8, -18.4,
          -19.4, -20.8, -19.9, -20.5, -16.6, -18.8, -15.6, -14.43, -16.2, -17.3, -18.6, -17.2, -16.32, -13.2,
          -15.4, -19.2, -19.4, -18.2, -18.1, -17.6, -15.8, -19.4, -18.5, -17.3, -18.7, -18.6, -19.6, -19.4, -15.2, -14.4
          ]

x = [_ for _ in range(len(reward))]

plt.figure(figsize=(15, 10))  # 设置画布的尺寸
plt.title('Fitness——Epochs', fontsize=20)  # 标题，并设定字号大小
plt.xlabel('Epochs', fontsize=14)  # 设置x轴，并设定字号大小
plt.ylabel('Fitness', fontsize=14)  # 设置y轴，并设定字号大小
plt.plot(x, reward, marker='*', linestyle='-', linewidth=2, label='Fitness')
plt.legend(loc=2, frameon=False, fontsize='xx-large')  # 图例展示位置，数字代表第几象限
plt.show()
