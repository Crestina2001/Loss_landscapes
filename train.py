import tqdm
import time
from utils import Accumulator, accuracy, generate_root, plt_and_save
from backbone import torch, nn
import torch.optim as optim
from configs import model_sv_rt, swa_start
import numpy as np

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# evaluate the performance of the model on testset
def eval(net, iter):
    net.to(device)  # move to gpu
    # prepare to count predictions for each class
    correct = 0
    total = 0
    with torch.no_grad():
        for data in iter:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / (total + 0.00001)  # in case total is 0


# train the model
def train(net, trainloader, testloader, optimizer, criterion, num_epoch, sv_mid=False):
    Loss, trainAcc, testAcc = [], [], []
    net.to(device)  # move the net to cuda
    sv_itv = max(num_epoch // 10, 1)  # save intervals
    ROOT = generate_root(model_sv_rt)  # root of trained models
    beg_time = time.time()
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        metric = Accumulator(3)  # Save the loss and accuracy in each batch
        # progress bar
        with tqdm.tqdm(total=len(trainloader)) as pbar:
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(loss.item() * labels.shape[0], accuracy(outputs, labels), inputs.shape[0])
                pbar.update(1)

        # compute the loss
        running_loss = metric[0] / metric[2]
        print(f'[{epoch + 1}] loss: {running_loss:.3f}')
        Loss.append(running_loss)

        # compute the training accuracy
        train_acc = metric[1] / metric[2]
        print(f'[{epoch + 1}] training accuracy: {train_acc:.3f}')
        trainAcc.append(train_acc)

        # compute the testing accuracy
        test_acc = eval(net, testloader)
        testAcc.append(test_acc)
        print(f'[{epoch + 1}] testing accuracy: {test_acc:.3f}')

        # save the model
        if epoch % sv_itv == 0 and sv_mid:
            PATH = ROOT + '/ep' + str(epoch + 1) + '.pth'
            # print(PATH)
            torch.save(net.state_dict(), PATH)
            print(f'epoch {epoch + 1} model saved !')

    end_time = time.time()
    print(f'Finished Training In {end_time - beg_time: .3f} s')
    # save the final model
    PATH = ROOT + '/' + 'final' + '.pth'
    torch.save(net.state_dict(), PATH)
    # plot loss and accuracy
    plt_and_save(Loss, trainAcc, testAcc, ROOT)


# train the model using swa
def train_swa(net, trainloader, testloader, optimizer, criterion, num_epoch, sv_mid=False):
    # swa parameters
    swa_model = AveragedModel(net).to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    Loss, trainAcc, testAcc, lrs = [], [], [], []
    net.to(device)  # move the net to cuda
    sv_itv = max(num_epoch // 10, 1)  # save intervals
    ROOT = generate_root(model_sv_rt)  # root of trained models
    beg_time = time.time()
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        metric = Accumulator(3)  # Save the loss and accuracy in each batch
        # progress bar
        with tqdm.tqdm(total=len(trainloader)) as pbar:
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # swa
                if epoch > swa_start:
                    swa_model.update_parameters(net)
                    swa_scheduler.step()
                else:
                    scheduler.step()

                with torch.no_grad():
                    metric.add(loss.item() * labels.shape[0], accuracy(outputs, labels), inputs.shape[0])
                pbar.update(1)
        # print the learning rates
        lr = optimizer.param_groups[0]['lr']
        print(f'[{epoch + 1}] learning rate: {lr:.7f}')
        lrs.append(lr)

        # compute the loss
        running_loss = metric[0] / metric[2]
        print(f'[{epoch + 1}] loss: {running_loss:.3f}')
        Loss.append(running_loss)

        # compute the training accuracy
        train_acc = metric[1] / metric[2]
        print(f'[{epoch + 1}] training accuracy: {train_acc:.3f}')
        trainAcc.append(train_acc)

        # compute the testing accuracy
        if epoch > swa_start:
            test_acc = eval(swa_model, testloader)
        else:
            test_acc = eval(net, testloader)
        testAcc.append(test_acc)
        print(f'[{epoch + 1}] testing accuracy: {test_acc:.3f}')

        # save the model
        if epoch % sv_itv == 0 and sv_mid:
            PATH = ROOT + '/ep' + str(epoch + 1) + '.pth'
            # print(PATH)
            torch.save(swa_model.state_dict(), PATH)
            print(f'epoch {epoch + 1} model saved !')

    end_time = time.time()
    print(f'Finished Training In {end_time - beg_time: .3f} s')
    # save the final model
    PATH = ROOT + '/' + 'final' + '.pth'
    torch.save(swa_model.state_dict(), PATH)
    # plot loss and accuracy
    plt_and_save(Loss, trainAcc, testAcc, lrs, ROOT)

