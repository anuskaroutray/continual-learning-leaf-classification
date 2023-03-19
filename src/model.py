import torch
import torch.nn as nn

from torchvision.models import resnet50, vgg19

class BaselineModel(nn.Module):

    def __init__(self, model_name, pretrained = True, num_classes = 30):
        super(BaselineModel, self).__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes

        if self.model_name == "resnet50":
            self.model = resnet50(pretrained = pretrained)
            self.model.fc = nn.Linear(2048, self.num_classes, bias = True)

        elif self.model_name == "inceptionv3":
            raise NotImplementedError

        elif self.model_name == "vgg19":
            self.model = vgg19(pretrained = pretrained)
            self.model.classifier[6] = nn.Linear(4096, self.num_classes, bias = True)
        
        elif self.model_name == "densenet":
            raise NotImplementedError

        # NOTE: Implement some more layers here later
        
    
    def forward(self, x):

        return self.model(x)