from torch import nn
import torch

# TODO add custom models


def get_custom_4_layer():
    pass


def get_cnn(arch, num_classes, family, pretrained=True):
    model = arch(pretrained=pretrained)
    if family == 'densenet':
        temp = model.classifier.in_features
        model.classifier = nn.Linear(temp, num_classes)
    else:
        temp = model.fc.in_features
        model.fc = nn.Linear(temp, num_classes)
    return model


def get_resnet(arch, num_classes, pretrained=True):
    return get_cnn(arch, num_classes, 'resnet', pretrained)


def get_densenet(arch, num_classes, pretrained=True):
    return get_cnn(arch, num_classes, 'densenet', pretrained)


def get_vgg(arch, num_classes, pretrained=True):
    model = arch(pretrained=pretrained)
    temp = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(temp, num_classes)
    return model


def get_cnn_features(arch, num_classes, pretrained=True):
    # TODO SVM, logistic regression
    # TODO SVM ridge operators
    # VGG for this
    model = arch(pretrained=pretrained)
    children = list(model.children())
    return nn.Sequential(*children[:-1])


class Ensemble(torch.nn.Module):
    def __init__(self, models, weights, feature_dims=512, num_classes=3):
        super().__init__()
        self.model1 = models[0]
        self.model2 = models[1]
        self.model3 = models[2]
        self.model1.requires_grad_(False)
        self.model2.requires_grad_(False)
        self.model3.requires_grad_(False)
        # self.model1.load_state_dict(torch.load(weights[0]))
        # self.model2.load_state_dict(torch.load(weights[1]))
        # self.model3.load_state_dict(torch.load(weights[2]))
        model1_infeatures = self.model1.classifier[6].in_features
        model2_infeatures = self.model2.fc.in_features
        model3_infeatures = self.model3.classifier.in_features
        feature_dims = min(model1_infeatures, model2_infeatures, model3_infeatures, feature_dims)
        self.model1.classifier[6] = nn.Linear(model1_infeatures, feature_dims) # vgg16
        self.model2.fc = nn.Linear(model2_infeatures, feature_dims)
        self.model3.classifier = nn.Linear(model3_infeatures, feature_dims) #densenet161
        self.finalFc = nn.Linear(feature_dims, num_classes)

    def forward(self, xb):
        out1 = self.model1(xb)
        out2 = self.model2(xb)
        out3 = self.model3(xb)
        out = out1 + out2 + out3
        return self.finalFc(out)

    def unfreeze(self):
        self.model1.requires_grad_(True)
        self.model2.requires_grad_(True)
        self.model3.requires_grad_(True)
        self.finalFc.requires_grad_(True)
