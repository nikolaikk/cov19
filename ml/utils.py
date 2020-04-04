import torch
from PIL import Image
import torchvision.transforms.functional as F
import torch

def inference(image_path, model):

    # set device
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    image = Image.open(image_path)

    x = F.resize(image, 224)
    x = F.center_crop(x, 224)
    x = F.to_tensor(x).to(device)
    x.unsqueeze_(0)

    y_pred = model(x)
    y_pred = torch.nn.functional.softmax(y_pred)
    y_pred = y_pred.cpu().detach().numpy()
    return y_pred
