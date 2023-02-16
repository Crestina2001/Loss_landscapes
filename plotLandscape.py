import torch
import copy
import numpy as np
import matplotlib.pyplot as plt

# Automatically generate a new folder under the sv_rt
def generate_root(sv_rt):
    PATH = sv_rt
    dir = os.listdir(PATH)
    name_idx = 1
    while True:
        name = 'ex' + str(name_idx)
        filePath = PATH + '/' + name
        if os.path.exists(filePath):
            name_idx += 1
            continue
        PATH += '/' + name
        os.mkdir(PATH)
        return PATH

# Plot 1-D landscapes of random directions
class Drawing:
    '''
    net: your neural network class
    path2weights: the path to the pth file
    '''
    def __init__(self, net, path2weights):
        net.load_state_dict(torch.load(path2weights))
        net.eval()
        self.net = net
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weights = self.get_weights(self.net)

    # Extract parameters from net, and return a list of tensors
    def get_weights(self, net):
        return [p.data for p in net.parameters()]

    def get_random_weights(self):
        return [torch.randn(w.size()) for w in self.weights]

    # Filter-wise rescale the direction
    def normalize_directions(self, direction, ignore='biasbn'):
        assert (len(direction) == len(self.weights))
        for d, w in zip(direction, self.weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0)  # ignore directions for weights with 1 dimension
                else:
                    d.copy_(w)  # keep directions for weights/bias that are only 1 per node
            else:
                for d, w in zip(direction, self.weights):
                    d.mul_(w.norm() / (d.norm() + 1e-10))

    # Setup a target direction from one model to the other
    def create_random_direction(self, ignore='biasbn'):
        direction = self.get_random_weights()
        self.normalize_directions(direction, ignore)

        return direction

    # set self.net to: (original network parameters) + alpha * directions
    def set_weights(self, directions, alpha):
        for (p, w, d) in zip(self.net.parameters(), self.weights, directions):
            p.data = w + alpha * torch.Tensor(d).type(type(w))

    # evaluate the performance(accuracy and loss) of the model on trainset or testset
    def eval(self, iter, criterion=None):
        # prepare to count predictions for each class
        correct = 0
        total = 0
        loss = 0
        self.net.to(self.device)
        with torch.no_grad():
            for i, data in enumerate(iter, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if criterion:
                    loss += labels.shape[0] * criterion(outputs, labels).item()
            acc = correct / (total + 0.00001)  # in case total is 0
            loss = loss / (total + 0.00001)
            if criterion:
                return loss, acc
            else:
                return acc

    '''
    Producing plots along random normalized directions
    Compulsory parameters:
        trainLoader: <class 'torch.utils.data.dataloader.DataLoader'>
    Optional parameters:
        testLoader: <class 'torch.utils.data.dataloader.DataLoader'>
        criterion: loss functions like nn.CrossEntropyLoss()
        num_plot: the number of generating random directions
        num_point: the number of sampling points of alpha
        sv_rt: a string pointing to the folder where plots will be saved
    '''
    def plt_random_normalized_dir(self, trainLoader, testLoader=None,
                                  criterion=None, num_plot=3, num_point=50, sv_rt='./Drawings/'):
        alphas = np.linspace(-1, 1, num_point)
        #print(alphas)
        acc_collections = []
        for i in range(num_plot):
            random_dir = self.create_random_direction()
            self.set_weights(random_dir, -1.0)
            acc_oneRun = []
            print(f'Plot {i + 1}:')
            for alpha in alphas:
                train_loss, train_acc, test_loss, test_acc = [], [], [], []
                self.set_weights(random_dir, alpha)
                if criterion:
                    train_loss, train_acc = self.eval(trainLoader, criterion)
                else:
                    train_acc = self.eval(trainLoader)
                if testLoader:
                    if criterion:
                        test_loss, test_acc = self.eval(testLoader, criterion)
                    else:
                        test_acc = self.eval(testLoader)
                print(f'Alpha: {alpha:.3f}, Train accuracy: {train_acc:.3f}')
                acc_oneRun.append([train_loss, train_acc, test_loss, test_acc])
            acc_collections.append(np.array(acc_oneRun).transpose())
        #print(acc_collections)
        # plot the figure
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("alpha")
        ax2 = ax1.twinx()
        for i in range(num_plot):
            if criterion:
                # train loss
                ax1.plot(alphas, acc_collections[i][0], 'b')
                if testLoader:
                    # test loss
                    ax1.plot(alphas, acc_collections[i][2],'b--')
            # train accuracy
            ax2.plot(alphas, acc_collections[i][1], 'r')
            if testLoader:
                # test accuracy
                ax2.plot(alphas, acc_collections[i][3], 'r--')
        PATH = generate_root(sv_rt) +  '/' + 'Plot.png'
        fig.savefig(PATH)



