import torch
from torch.utils.data import DataLoader
from dataloader import CaltechDataset
from model import MyNet
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models

import matplotlib.pyplot as plt
from plot_weights import show_filts


if __name__ == "__main__":
    device = 'cuda:0'
    epoch_size = 10
    batch_size = 15
    learning_rate = 1e-3

    data = CaltechDataset("data")
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train, validation = torch.utils.data.random_split(data, [train_size, test_size],
                                                      generator=torch.default_generator.manual_seed(42))
    validation.augment = False

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)

    model = models.resnet152(pretrained=True).to(device)
    state_dict = torch.load('weights/weights_9')

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 257).to(device)

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.load_state_dict(state_dict)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_size):
        for x, y in tqdm(train_loader):
            model = model.train()
            train = x.to(device)
            label = y.to(device)

            y_pred = model(train)
            loss = loss_fn(y_pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        score = 0

        for x, y in tqdm(valid_loader):
            model = model.eval()
            with torch.no_grad():
                valid = x.to(device)
                label = y.to(device)

                y_pred = model(valid)

                m = F.softmax(y_pred, dim=1)
                y_pred_value = torch.argmax(m, 1)

                score += (y_pred_value == label).sum().item()

        print("epoch " + str(epoch + 1) + ":" + str(score / len(validation)))
        torch.save(model.state_dict(), 'weights/weights_{}'.format(epoch))

    torch.save(model.state_dict(), 'weights/weights_final')

