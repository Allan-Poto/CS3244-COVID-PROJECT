import io
import tensorflow as tf
import matplotlib.pyplot as plt
import scikitplot as skplt
import itertools
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import random
from . import DATA_DIR, IMAGE_DIR, PHASE_COL
from .xray_dataset import XrayDataset

SEED = 0

dataset1_path = DATA_DIR+'covid-xrays/data.csv'
dataset2_path = DATA_DIR+'pneumonia_1/data.csv'
dataset3_path = DATA_DIR+'covid_2/data.csv'

# Dataset from:
# https://www.kaggle.com/tawsifurrahman/covid19-radiography-database


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    figure = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names, rotation=90)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center", fontsize=16,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center", fontsize=16,
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    return figure


def plot_roc(labels, preds):
    """
        labels must be integer, preds must be probabilities
    """
    skplt.metrics.plot_roc_curve(
        labels, preds, title='ROC curves on test data (trained)', curves=['each_class'])
    plt.show()


def class_weights(labels):
    weight = []
    big_i = labels.max().item() + 1
    for i in range(big_i):
        weight.append((labels == i).sum())
    weight = torch.tensor(weight, dtype=torch.float32)
    return len(labels)/weight


def get_weighted_sampler(label_list):
    weights = class_weights(label_list)
    weights_as_idx = weights[label_list]
    weighted_sampler = WeightedRandomSampler(
        weights=weights_as_idx,
        num_samples=len(weights_as_idx),
        replacement=True
    )
    return weighted_sampler


def to_dataloader(ds, bs=256, weighted=False, train=False):
    if weighted and len(ds) > 0:
        labels = torch.stack(ds.labels)
        return DataLoader(ds, num_workers=torch.cuda.device_count()*4,
                          #   shuffle=train,
                          sampler=get_weighted_sampler(labels),
                          drop_last=False,
                          batch_size=bs)

    dl = DataLoader(ds,
                    num_workers=torch.cuda.device_count()*4,
                    # shuffle=train,
                    drop_last=False,
                    batch_size=bs)
    return dl


def get_covid_dls(bs, df, filecol, labelcol, root=IMAGE_DIR, reg=True):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    phase = ['train', 'val', 'test']
    dataloaders = []
    for p in phase:
        # filters out dataframe based on phase
        phase_df = df[df[PHASE_COL] == p]
        temp_ds = XrayDataset(root, filecol, labelcol,
                              phase_df, train=(p == 'train' and reg))
        dataloaders.append(to_dataloader(
            temp_ds,
            bs=bs,
            weighted=(p == 'train')))
    return dataloaders


def split(df, val_pct=0.2, test_pct=0.2):
    torch.manual_seed(0)
    rand_indices = torch.randperm(len(df))
    num_val = int(val_pct * len(df))
    num_test = int(test_pct * len(df))
    num_train = len(df) - num_val - num_test
    df['phase'] = ''
    df['phase'].iloc[rand_indices[0:num_train]] = 'train'
    df['phase'].iloc[rand_indices[num_train:num_train+num_val]] = 'val'
    df['phase'].iloc[rand_indices[num_train+num_val:]] = 'test'
    return df
