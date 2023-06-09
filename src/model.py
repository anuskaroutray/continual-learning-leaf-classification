import torch
import torch.nn as nn

from torchvision.models import resnet50, vgg19, inception_v3, densenet121, googlenet

class BaselineModel(nn.Module):

    def __init__(self, model_name, pretrained = True, num_classes = 30):
        super(BaselineModel, self).__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes

        if self.model_name == "resnet50":
            self.model = resnet50(pretrained = self.pretrained)
            self.model.fc = nn.Linear(2048, self.num_classes, bias = True)

        elif self.model_name == "inceptionv3":
            self.model = googlenet(pretrained = self.pretrained)
            self.model.fc = nn.Linear(1024, self.num_classes, bias = True)

        elif self.model_name == "vgg19":
            self.model = vgg19(pretrained = self.pretrained)
            self.model.classifier[6] = nn.Linear(4096, self.num_classes, bias = True)
        
        elif self.model_name == "densenet":
            self.model = densenet121(pretrained = self.pretrained)
            self.model.classifier = nn.Linear(1024, self.num_classes, bias=True)

    def forward(self, x):
        return self.model(x)