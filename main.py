# import argparse
import os
import torch
from tensorboardX import SummaryWriter
from arch import all_models
from tqdm.auto import tqdm  # auto adjust to notebook and terminal
from data.utils import get_covid_dls, plot_confusion_matrix, plot_to_image, SEED
from utils import all_loss, all_opt
from utils.functions import accuracy
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score, f1_score, confusion_matrix
import numpy as np
import random

DEVICE_COUNT = torch.cuda.device_count()
IS_CUDA = torch.cuda.is_available()
LOG_DIR = os.path.join('logs')
WEIGHTS_DIR = os.path.join('model_weights')

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


DEFAULT_HP = {
    'epochs': 5,  # number of times we're training on entire dataset
    'loss': 'cross_entropy',
    'opt': 'SGD',
    'wd': 0.001,
    'lr': 3e-3
}


class Trainer():
    def __init__(self, exp_name, model, dls, hp, weights=None, sched=True):
        self.model = all_models[model]()
        self.device = torch.device("cuda" if IS_CUDA else "cpu")
        opt = all_opt[hp['opt']]
        if hp['opt'] == 'ADAM':
            self.opt = opt(
                params=self.model.parameters(),
                lr=hp['lr'],
                weight_decay=hp['wd']
            )
        else:
            self.opt = opt(
                params=self.model.parameters(),
                lr=hp['lr'],
                momentum=0.9,
                weight_decay=hp['wd']
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=2, gamma=0.1 if sched else 1)
        if weights is not None:
            weights.to(self.device)
        self.loss = all_loss[hp['loss']](weights)  # create all_loss dictionary
        self.epochs = hp['epochs']
        self.model.to(self.device)
        self.writer = SummaryWriter(os.path.join(
            LOG_DIR, exp_name, hp['model_name']))
        self.writer_val = SummaryWriter(os.path.join(
            LOG_DIR, exp_name, f"{hp['model_name']}_val"))
        self.exp_name = exp_name
        self.model_name = hp['model_name']
        self.hp = hp
        self.val_loss = []
        self.trng_loss = []
        self.test_loss = [0]
        self.val_acc = []
        self.trng_acc = []
        self.test_acc = [0]
        train, val, test = dls
        self.class_names = train.dataset.class_names
        self.train_dl = train
        self.val_dl = val
        self.test_dl = test
        self.steps = [0, 0]
        self.cms = {0: None, 1: None, 2: None}
        self.auc = {0: None, 1: None, 2: 0}
        self.recall = {0: None, 1: None, 2: [0 for i in (self.class_names)]}

    def freeze(self):
        self.unfreeze(-1)

    def unfreeze(self, n):
        modules = list(trainer.model.children())
        for m in modules:
            m.requires_grad_(True)
        for m in modules[:n]:
            m.requires_grad_(False)

    def train(self):
        losses = []
        # it turns dropout
        self.model.train()
        acc_count, data_count = 0, 0
        # we use tqdm to provide visual feedback on training
        for xb, yb in tqdm(self.train_dl, total=len(self.train_dl)):
            xb = xb.to(self.device)  # BATCH_SIZE, 3, 224, 224
            yb = yb.to(self.device)  # BATCH_SIZE, 1
            self.opt.zero_grad()
            output = self.model(xb)  # BATCH_SIZE, 3
            acc_count += accuracy(output, yb)
            data_count += yb.shape[0]  # in the event of drop last
            loss = self.loss(output, yb)
            loss.backward()  # calculates gradient descent
            self.opt.step()  # updates model parameters
            losses.append(loss)
            self.steps[0] += 1
            self._log('train_loss', loss, self.steps[0])
        losses = torch.stack(losses)
        epoch_loss = losses.mean().item()
        epoch_acc = acc_count/data_count
        self.trng_loss.append(epoch_loss)
        self.trng_acc.append(epoch_acc)
        print("\nepoch trng info: loss:{}, acc:{}".format(epoch_loss, epoch_acc))

    def validate(self):
        losses = []
        acc_count, data_count = 0,  0
        with torch.no_grad():  # don't accumulate gradients, faster processing
            self.model.eval()  # ignore dropouts and weight decay
            for xb, yb in tqdm(self.val_dl, total=len(self.val_dl)):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                output = self.model(xb)
                acc_count += accuracy(output, yb)
                data_count += yb.shape[0]  # in the event of drop last
                loss = self.loss(output, yb)
                losses.append(loss)
                self.steps[1] += 1
                self._log('val_loss', loss, self.steps[1], writer=1)
        losses = torch.stack(losses)
        epoch_loss = losses.mean().item()
        epoch_acc = acc_count/data_count
        self.val_loss.append(epoch_loss)
        self.val_acc.append(epoch_acc)
        print("\nepoch val info: loss:{}, acc:{}".format(epoch_loss, epoch_acc))

    def test(self):
        losses = []
        acc_count, data_count = 0,  0
        with torch.no_grad():  # don't accumulate gradients, faster processing
            self.model.eval()  # ignore dropouts and weight decay
            for xb, yb in tqdm(self.test_dl, total=len(self.test_dl)):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                output = self.model(xb)
                acc_count += accuracy(output, yb)
                data_count += yb.shape[0]  # in the event of drop last
                loss = self.loss(output, yb)
                losses.append(loss)
        if len(losses) > 0:
            losses = torch.stack(losses)
            epoch_loss = losses.mean().item()
            epoch_acc = acc_count/data_count
            self.test_loss.append(epoch_loss)
            self.test_acc.append(epoch_acc)
            print("\nepoch test info: loss:{}, acc:{}".format(epoch_loss, epoch_acc))

    def one_cycle(self):
        for i in range(self.epochs):
            print("epoch number: {}".format(i))
            self.train()
            self.validate()
            self.scheduler.step()
            self._save_weights()
        self.test()
        self._write_hp()  # for comparing between experiments

    def _log(self, phase, value, i, writer=0):
        if writer == 0:
            self.writer.add_scalar(
                tag=phase, scalar_value=value, global_step=i)
        else:
            self.writer_val.add_scalar(
                tag=phase, scalar_value=value, global_step=i)

    def _write_hp(self):
        val_loss, val_acc = min(self.val_loss), max(self.val_acc)
        test_loss, test_acc = min(self.test_loss), max(self.test_acc)

        self.confusion_matrix_auc(1)
        if len(self.test_dl) > 0:
            self.confusion_matrix_auc(2)
        metric_names = ['val_loss', 'val_acc',
                        'test_loss', 'test_acc',
                        'val_auc', 'test_auc',
                        'val_recall', 'test_recall']
        metric_values = [val_loss, val_acc, test_loss,
                         test_acc, self.auc[1], self.auc[2],
                         self.recall[1][0], self.recall[2][0]]
        metrics = dict(zip(metric_names, metric_values))
        self.writer.add_hparams(self.hp, metrics)

    def load_weights(self, pkl_name, num_classes=None, family=None):
        weights_path = os.path.join(
            WEIGHTS_DIR, self.exp_name, pkl_name)
        sd = torch.load(weights_path)
        self.model.load_state_dict(sd)
        if num_classes is not None and family is not None:
            if family == 'densenet':
                temp = self.model.classifier.in_features
                self.model.classifier = torch.nn.Linear(temp, num_classes)
            else:
                temp = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(temp, num_classes)
        self.model.to(self.device)

    def _save_weights(self):
        best_val = min(self.val_loss)
        if self.val_loss[-1] == best_val:
            weights_path = os.path.join(
                WEIGHTS_DIR, self.exp_name, self.model_name+'.pkl')
            os.makedirs(os.path.join(
                WEIGHTS_DIR, self.exp_name), exist_ok=True)
            self.model.cpu()
            state = self.model.state_dict()
            torch.save(state, weights_path)  # open(pkl), compress
            self.model.to(self.device)

    def confusion_matrix_auc(self, dl_type):
        if self.cms[dl_type] is not None and self.auc[dl_type] is not None:
            return self.cms[dl_type], self.auc[dl_type]
        dls = [self.train_dl, self.val_dl, self.test_dl]
        dl = dls[dl_type]
        preds_raw = torch.tensor([])
        labels_int = torch.tensor([], dtype=torch.long)
        with torch.no_grad():
            for batch in dl:
                images, labels = batch
                labels_int = torch.cat((labels_int, labels), dim=0)
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                preds_raw = torch.cat(
                    (preds_raw, preds.cpu()), dim=0
                )
        preds_int = torch.argmax(preds_raw, dim=1)
        cm = confusion_matrix(labels_int, preds_int)
        self.cms[dl_type] = cm
        # recall and auc
        self.recall[dl_type] = recall_score(
            labels_int, preds_int, average=None)
        labels_one_hot = torch.nn.functional.one_hot(
            labels_int, num_classes=len(self.class_names))
        preds_one_hot = torch.nn.functional.one_hot(
            preds_int, num_classes=len(self.class_names))
        print('all_recall', self.recall[dl_type])
        print('f1_score', f1_score(labels_one_hot,
                                   preds_one_hot, average="weighted"))
        print('weighted_recall', recall_score(
            labels_one_hot, preds_one_hot, average='weighted'))
        covid_one_hot_labels = labels_one_hot[:, 0]
        covid_one_hot_preds = preds_one_hot[:, 0]
        fpr, tpr, thresholds = roc_curve(
            covid_one_hot_labels, covid_one_hot_preds)
        self.auc[dl_type] = auc(fpr, tpr, reorder=True)
        return preds_raw, labels_int


if __name__ == "__main__":
    dls = get_covid_dls(bs=256)
    model = 'resnet18'
    hp = {**DEFAULT_HP, 'model_name': model+'_0'}
    trainer = Trainer(model, dls, hp)
