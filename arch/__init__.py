from torchvision import models
import torch
from .cnn import *

NUM_CLASSES = 3


all_models = {
    'resnet18': lambda: get_resnet(models.resnet18, NUM_CLASSES, pretrained=True),
    'resnet34': lambda: get_resnet(models.resnet34, NUM_CLASSES, pretrained=True),
    'resnet50': lambda: get_resnet(models.resnet50, NUM_CLASSES, pretrained=True),
    'resnet101': lambda: get_resnet(models.resnet101, NUM_CLASSES, pretrained=True),
    'densenet121': lambda: get_densenet(models.densenet121, NUM_CLASSES, pretrained=True),
    'densenet161': lambda: get_densenet(models.densenet161, NUM_CLASSES, pretrained=True),
    'densenet201': lambda: get_densenet(models.densenet201, NUM_CLASSES, pretrained=True),
    'vgg16bn': lambda: get_vgg(models.vgg16_bn, NUM_CLASSES, pretrained=True),
    'vgg16': lambda: get_vgg(models.vgg16, NUM_CLASSES, pretrained=True),
}

r34 = all_models['resnet34']()
vgg16 = all_models['vgg16']()
d121 = all_models['densenet121']()
vgg16_weights = './model_weights/report/vgg16_224_reg_4.pkl'
r34_weights = './model_weights/report_all/resnet34_224_0.pkl'
d121_weights = './model_weights/report_all/densenet121_224_0.pkl'
ENSEMBLE_WEIGHTS = [vgg16_weights, r34_weights, d121_weights]
vgg16.load_state_dict(torch.load(ENSEMBLE_WEIGHTS[0]))
r34.load_state_dict(torch.load(ENSEMBLE_WEIGHTS[1]))
d121.load_state_dict(torch.load(ENSEMBLE_WEIGHTS[2]))
ENSEMBLE_MODELS = [vgg16, r34, d121]
all_models['ensemble'] = lambda: Ensemble(ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS, 512, 3)
