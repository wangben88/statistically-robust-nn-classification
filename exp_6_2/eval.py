from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

import torch.distributions as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from exp_6_2.network import WideResNet

criterion = nn.CrossEntropyLoss()
x_min = torch.tensor([(0.0 - 0.4914)/0.2023, (0.0 - 0.4822)/0.1994, (0.0 - 0.4465)/0.2010]).to(device)
x_max = torch.tensor([(1.0 - 0.4914)/0.2023, (1.0 - 0.4822)/0.1994, (1.0 - 0.4465)/0.2010]).to(device)

def pgd_linf(model, X, y, epsilon, alpha, num_iter=7, randomize=False):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = torch.max(delta.data, x_min.view([1, -1, 1, 1]) - X)
        delta.data = torch.min(delta.data, x_max.view([1, -1, 1, 1]) - X)
        delta.grad.zero_()
    return delta.detach()

def compute_metric(loader, net, epsilon, adv):
    """ Calculate A-TSRM or adversarial accuracy"""
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs_pert = inputs
        if adv:
            inputs_pert += pgd_linf(net, inputs, targets, epsilon, epsilon/5)
        else:
            inputs_dist = dist.Uniform(low=torch.max(inputs - epsilon, x_min.view([1, -1, 1, 1])), high=torch.min(inputs + epsilon, x_max.view([1, -1, 1, 1])))
            inputs_pert = inputs_dist.sample().to(device)
        targets_pert = targets
        targets_pert_pred = net(inputs_pert)
        
        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()

    acc = 100.*correct/total
    return(acc)
    

def eval_metric(modelfilename, eval_epsilons, adv=False, train=False):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=50, shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=50, shuffle=False)
    
    # Load model
    model = WideResNet(28, 10, 0.3, 10)
    if device == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    model.load_state_dict(torch.load(modelfilename)["net"])

    accs = []
    for eval_epsilon in eval_epsilons:
        print("Epsilon: ", eval_epsilon)
        if train:
            acc = compute_metric(train_loader, model, eval_epsilon, adv)
        else:
            acc = compute_metric(test_loader, model, eval_epsilon, adv)
        print(acc)
        accs.append(acc)

    return accs

if __name__ == '__main__':
    # Read command arguments
    parser = argparse.ArgumentParser(description='Computation of TSRM (total statistical robustness metric)')
    parser.add_argument('--model_loc', help='location of saved model')
    parser.add_argument('--prefix', help='prefix for saving results')
    args = parser.parse_args()

    # ./data/ckpt_stat_epsilon_{args.epsilon}_run_{args.run}
    modelfilename = os.path.join(args.model_loc, args.prefix+'.pth')
    eval_epsilons = [0, 0.157, 0.5]
    eval_metric(modelfilename, eval_epsilons)
