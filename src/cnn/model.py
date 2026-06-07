import torch.nn as nn
from torchvision import models

NUM_CLASSES = 5

def build_model():

    model = models.densenet121(
        weights=None
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model