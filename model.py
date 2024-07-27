import torch
import torch.nn as nn
import torchvision.models as models

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=3):
        super(EmotionRecognitionModel, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)