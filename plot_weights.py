import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
import matplotlib.pyplot as plt
import math


def show_filts(model, n_cols=32, figsize=(32,2), global_scale=True):
    filts = list(model.parameters())[0]
    filts = filts.detach().cpu().numpy().transpose(0, 2, 3, 1)
    n_filts = filts.shape[0]

    n_rows = math.ceil(n_filts/n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax = axs[i]
        if i < n_filts:
            filt = filts[i]
            ax.axis('off')
            if global_scale:
                filt = (filt - filts.min())/max(filts.max() - filts.min(), 1e-16)
            else:
                filt = (filt - filt.min())/max(filt.max() - filt.min(), 1e-16)
            ax.imshow(filt)
        else:
            fig.delaxes(ax)

if __name__ == '__main__':
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 257)

    state_dict = torch.load('weights/weights_15')
    model.load_state_dict(state_dict)

    plt.figure()
    show_filts(model)
    plt.show()
