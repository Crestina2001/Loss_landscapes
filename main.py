import argparse
from backbone import LeNet, AlexNet, VggNet, ResNet, MyNet, nn
from utils import get_CIFAR10
from configs import batch_size, learning_rate, num_epoch, sv_mid, is_plot, PATH2MODEL
# testing the torchvision models
import torchvision.models as models
from train import train, train_swa
import torch.optim as optim
import torch
from torch.optim.swa_utils import AveragedModel

from plotLandscape import Drawing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--swa')
    args = parser.parse_args()

    model = args.model
    trainWithSwa = False
    if args.swa != None:
        trainWithSwa = True
    trainloader, testloader = get_CIFAR10(bs=batch_size, Resize=False)

    # use cross entropy loss funcion
    criterion = nn.CrossEntropyLoss()

    net = None
    if model == 'lenet':
        net = LeNet()
    elif model == 'alexnet':
        net = AlexNet()
    elif model == 'vggnet':
        net = VggNet()
    elif model == 'resnet':
        net = ResNet()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    if is_plot:
        # net = AveragedModel(net)
        drawing = Drawing(net, PATH2MODEL)
        drawing.plt_random_normalized_dir(trainloader, testLoader=testloader, criterion=criterion)

    else:
        if args.swa != None:
            train_swa(net, trainloader, testloader, optimizer, criterion, num_epoch, sv_mid)
        else:
            train(net, trainloader, testloader, optimizer, criterion, num_epoch, sv_mid)

