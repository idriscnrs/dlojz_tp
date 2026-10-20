import os
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


## Train dataset
transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),   # Horizontal Flip - Data Augmentation
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()                # Convert the PIL Image to a tensor
        ])
    
    
train_dataset = CIFAR10(root=os.environ['ALL_CCFRSCRATCH']+'/CIFAR_10',
                                             train=True, download=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset,    
                                           batch_size=4,
                                           shuffle=True)

def print_sample(dataloader):
    batch = next(iter(dataloader))
    print('X train batch, shape: {}, data type: {}, Memory usage: {} bytes'
          .format(batch[0].shape, batch[0].dtype, batch[0].element_size()*batch[0].nelement()))
    print('Y train batch, shape: {}, data type: {}, Memory usage: {} bytes'
          .format(batch[1].shape, batch[1].dtype, batch[1].element_size()*batch[1].nelement()))

    img = batch[0][0].numpy().transpose((1,2,0))
    fig, axes = plt.subplots(1, 10, figsize=(20, 2.4))
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0, hspace=0)  # zero margins so block 0 touches the figure's left edge
    ax = axes[0]  # digit in the leftmost 2in block
    ax.imshow(img)
    ax.set_title('label class: {}'.format(batch[1][0].numpy()))
    ax.axis('off')
    for a in axes[1:]: # remaining 11 blocks stay blank
        a.axis('off')
    plt.show()

## Val dataset
val_transform = transforms.Compose([transforms.ToTensor()])# convert the PIL Image to a tensor
    
val_dataset = CIFAR10(root=os.environ['ALL_CCFRSCRATCH']+'/CIFAR_10',
                                               train=False, download=False, transform=val_transform)
val_dataset

## Model
resnet18 = resnet18(num_classes=10)