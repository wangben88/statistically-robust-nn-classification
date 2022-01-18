# Run this file from the main directory as e.g.: python -m exp_6_1.train --epsilon=0.1 --epochs=50

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from exp_6_1.network import SimpleMlp

# Defines limits of MNIST elements
x_min = 0
x_max = 1

def run():
  # Load data
  #kwargs = {'num_workers': 1, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=100, shuffle=True)
  test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=100, shuffle=True)

  # Create model
  model = SimpleMlp()
  model.to(device)

  if args.resume:
    if args.cauchy:
      model.load_state_dict(torch.load(f'./data/mnist_simplemlp_stat_cauchy.pth'))
    else:
      model.load_state_dict(torch.load(f'./data/mnist_simplemlp_stat_{args.epsilon}.pth'))
  # Get an optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = torch.nn.CrossEntropyLoss()

  # Create the function to perform an epoch
  def train(epoch):
    model.train()
    total_loss = 0
    for _, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      data, target = data.float().to(device), target.long().to(device)
      data = data.view(-1, 784)
    
      if args.cauchy:
        data_dist = dist.Cauchy(loc = data, scale=torch.Tensor([0.5]).to(device))
      else:
        data_dist = dist.Uniform(low=torch.max(data - args.epsilon, torch.Tensor([x_min]).to(device)), high=torch.min(data + args.epsilon, torch.Tensor([x_max]).to(device)))
      
      data_pert = data_dist.sample().to(device)
      data_pert = torch.clamp(data_pert, x_min, x_max)

      #print(data.size(), target.size())
    
      output = model(data_pert)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    total_loss /= len(train_loader.dataset)
    return total_loss

  def test(epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    for _, (data, target) in enumerate(test_loader):
      data, target = data.float().to(device), target.long().to(device)
      data = data.view(-1, 784)
    
      if args.cauchy:
        data_dist = dist.Cauchy(loc = data, scale=torch.Tensor([0.5]).to(device))
      else:
        data_dist = dist.Uniform(low=torch.max(data - args.epsilon, torch.Tensor([x_min]).to(device)), high=torch.min(data + args.epsilon, torch.Tensor([x_max]).to(device)))
      
      data_pert = data_dist.sample().to(device)
      data_pert = torch.clamp(data_pert, x_min, x_max)
    
      output = model(data_pert)
      preds = output.argmax(dim=1)
      total_correct += (preds == target).sum().float().item()
      loss = loss_fn(output, target)
      total_loss += loss.item()

    total_loss /= len(test_loader.dataset)
    total_correct /= len(test_loader.dataset)
    return total_loss, total_correct

  # Train the network
  print("Starting network training")
  for epoch in range(1, args.epochs + 1):
      train_loss = train(epoch)
      with torch.no_grad():
        test_loss, test_acc = test(epoch)

      print("[epoch %03d]  train loss: %.5f, test loss: %.5f, test acc: %.3f" % (epoch, train_loss, test_loss, test_acc))

  print("Training Done.")
  print(f"Final Loss: {test_loss}")

  if args.cauchy:
    torch.save(model.state_dict(), f'./data/mnist_simplemlp_stat_cauchy.pth')
  else:
    torch.save(model.state_dict(), f'./data/mnist_simplemlp_stat_{args.epsilon}.pth')

if __name__ == '__main__':
  # Fixing random seed for reproducibility
  seed = 0
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Read command arguments
  parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
  parser.add_argument('--epsilon', default = 0.3, type=float, help='perturbation radius')
  parser.add_argument('--resume', '-r', action='store_true', help='resume')
  parser.add_argument('--epochs', default=20, type=int, help='number of training epochs')
  parser.add_argument('--cauchy', '-c', action='store_true', help='use Cauchy distribution')
  args = parser.parse_args()

  # Train model
  run()
