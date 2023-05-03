import torch

all_opt = {
    'SGD': torch.optim.SGD,
    'ADAM': torch.optim.Adam
}


def class_weights(labels):
    weight = []
    big_i = labels.max().item() + 1
    for i in range(big_i):
        weight.append((labels == i).sum())
    weight = torch.tensor(weight, dtype=torch.float32)
    return len(labels)/weight


all_loss = {
    'cross_entropy': lambda w: torch.nn.CrossEntropyLoss(weight=w)
}


def accuracy(y_hat, y):
    preds = torch.argmax(y_hat, dim=1)
    return (preds == y).sum().item()
