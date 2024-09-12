import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import ConcatDataset 

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

#stack all train images together into a tensor of shape #(50000, 3, 32, 32) 
x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])

#get the mean of each channel
mean = torch.mean(x, dim=(0,2,3)) #tensor([0.4914, 0.4822, 0.4465])
std = torch.std(x, dim=(0,2,3)) #tensor([0.2470, 0.2435, 0.2616])

print("mean: ", mean)
print("std: ", std)

# The values obtained with mean and std is then given to transforms.Normalize()