import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(model, train_loader, loss_func, optimizer, epoch):

    correct = 0
    epoch_loss = 0
    epoch_counter = 0
    # switch model to train mode (dropout enabled)
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # send data to cuda
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1)[1]

        # Analyze the accuracy of the batch
        correct += (pred == target).long().sum().item()
        epoch_loss += loss.data.item() * data.shape[0]

        epoch_counter += float(data.shape[0])


        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


    return epoch_loss, correct, epoch_counter

def test(model, test_loader, loss_func):
    # switch model to eval model (dropout becomes pass through)

    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # send data to cuda
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += loss_func(output, target).item() * data.shape[0]
            pred = output.argmax(dim=1)
            correct += (pred == target).long().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.06f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct

def main():
    # seed pytorch random number generator for reproducablity
    torch.manual_seed(2)

    train_dataset = torchvision.datasets.CIFAR10(
        './data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        )

    test_dataset = torchvision.datasets.CIFAR10(
        './data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers=2)

    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5 ,padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Dropout2d(),
        nn.Conv2d(32, 32, kernel_size=5 ,padding=2),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, stride=2),
        nn.Conv2d(32, 64, kernel_size=5 ,padding=2),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(576, 64),
        nn.ReLU(),
        nn.Linear(64, 10))

    # send model parameters to cuda
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_func = nn.CrossEntropyLoss()

    epochs = 40
    train_accuracy = [0.1]*epochs
    train_loss = [0.1]*epochs
    test_accuracy = [0.1]*epochs
    test_loss_plot = [0.1]*epochs

    for epoch in range(epochs):

        train_loss, train_correct, epoch_counter = train(model, train_loader, loss_func, optimizer, epoch)

        # train_accuracy[epoch]=train_correct / epoch_counter*1.0
        # train_loss[epoch]=train_loss / epoch_counter*1.0

        test_loss, test_correct = test(model, test_loader, loss_func)

        test_accuracy[epoch]=test_correct / len(test_loader.dataset)
        test_loss_plot[epoch]=test_loss

    print('Saving model to CIFAR_10.pt')
    torch.save(model.state_dict(), 'CIFAR_10.pt')

    fig = plt.figure(figsize=(21, 7))
    plt.subplot(1,2,1)
    # plt.plot(train_loss, label='Training')
    plt.plot(test_loss_plot, label='Test')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Test Loss during Training', fontsize=16)
    plt.xticks(range(epochs))
    plt.grid('on')
    plt.legend(fontsize=14)

    plt.subplot(1,2,2)
    # plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.xticks(range(epochs))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Test Accuracy during Training', fontsize=16)
    plt.grid('on')

    fig.savefig('CIFAR_10_results.png')
    print('Saving image to CIFAR_10_results.png')

if __name__ == "__main__":
    main()
