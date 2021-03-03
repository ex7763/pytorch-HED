import torch
import yaml
import argparse 
import random
import numpy

from hed_pipeline import *
from attrdict import AttrDict
from models import HED

def save_model_with_jit():
    net = Network().eval()

    with torch.no_grad():
        traced_cell = torch.jit.trace(net, torch.FloatTensor(
            torch.rand([1, 3, 120, 160])))
    torch.jit.save(traced_cell, "pruned.pt")

def estimate(self, tenInput):
    if torch.cuda.is_available():
        tenInput = tenInput.cuda()

    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    return self.netNetwork(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()

def image_path_to_np(self, path):
    image = PIL.Image.open(path)
    image = image.convert('RGB')
    image = image.resize((160, 120))

    img = np.array(image)
    img = np.rollaxis(img, 2, 0)
    tenInput = torch.FloatTensor(img * (1.0 / 255.0))

    tenOutput = self.estimate(tenInput)

    tenOutput = tenOutput.clamp(
        0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0]

    return tenOutput.reshape((120, 160, 1))

    return tenOutput

    #image = PIL.Image.fromarray((tenOutput*255.0).astype(numpy.uint8))
    #image.save(out_dir + f + ".bmp")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    path = "/home/hpc/nctu/hed/ckpt/standard/log/new_github_adam_lr1e-4_Aug_0.3_savemodel_Feb25_12-13-51/epoch_0.pt"
    net = torch.jit.load(path)
    if torch.cuda.is_available():
        net = net.cuda().eval()
    else:
        net = net.eval()

    print(net)

    #img = torch.Tensor(torch.rand([1, 3, 120, 160]))
    img = torch.Tensor(torch.rand([1, 3, 240, 320])).to(device)

    result = net(img)

    print(result)
    """

    path = "/home/hpc/nctu/hed/ckpt/standard/log/new_github_adam_lr1e-4_Aug_0.3_savemodel_Feb25_12-13-51/epoch_0.pth"
    cfg = None
    with open('config/standard.yaml', 'r') as f:
        cfg = AttrDict( yaml.load(f) )

    net = HED.HED(cfg)
    net.load_state_dict(torch.load(path))
    print(net)
