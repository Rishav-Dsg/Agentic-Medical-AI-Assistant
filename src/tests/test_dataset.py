from cnn.dataset import ChestXrayDataset

from torchvision import transforms
from torch.utils.data import DataLoader

import torch 

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = ChestXrayDataset(
    csv_file="data/train.csv",
    image_dir="data/images",
    transform=train_transform
)

print("Dataset size:", len(dataset))

image, labels = dataset[0]

print(type(image))
if isinstance(image, torch.Tensor):
    print(image.shape)
print("Labels:", labels)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

images, labels = next(iter(loader))

print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)