import torch


def equal(prediction, label):
    predict = torch.argmax(prediction, dim=1)
    if torch.equal(predict, label):
        return 1
    else:
        return 0


def Accuracy(right_predict, all_predict):
    return all_predict / right_predict
