import torch
import torch.nn as nn
from dataloader import get_dataloader, get_dataset, get_transform
from model import ConvolutionNet
from trainer import test, train
from torch.utils.tensorboard import SummaryWriter
def main():
    transform = get_transform()
    train_dataset, valid_dataset, test_dataset = get_dataset(transform)
    train_loader, valid_loader, test_loader = get_dataloader(train_dataset, valid_dataset, test_dataset, 64)
    model = ConvolutionNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    writer = SummaryWriter()
    train(optimizer, train_loader, model, criterion, 32, valid_loader, writer)
    test(model, test_loader, writer)

if __name__ == '__main__':
    main()