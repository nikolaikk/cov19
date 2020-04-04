import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def resnet50():
    resnet50_path = os.path.join(os.path.dirname(__file__),'resnet50.pt')
    resnet = models.resnet50().to(device)
    resnet.fc = torch.nn.Linear(2048,4).to(device)

    # Eval mode
    resnet.eval()
    resnet.load_state_dict(torch.load(resnet50_path, map_location=torch.device(device)))

    return resnet
