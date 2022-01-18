# Run this file from the main directory as e.g.: python -m exp_6_3.train --epsilon=0.1 --epochs=50

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import pickle

import numpy as np
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from exp_6_3.network import SimpleMlp

# Defines limits of MNIST elements
x_min = 0
x_max = 1

def run():
  # Load data
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

  start_epoch = 0
  if args.resume:
    model.load_state_dict(torch.load(f'./data/mnist_simplemlp_stat_weighted_modeldict_sigma_{args.sigma}_run_{args.run}_epoch_1000.pth'))
    start_epoch = 1000

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  # Specify weighted cross entropy loss function
  loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 100, 1]).to(device))

  # Create the function to perform an epoch
  def train(epoch):
    model.train()
    total_loss = 0 # SRR loss
    total_nat_loss = 0 # Natural loss
    for _, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      data, target = data.float().to(device), target.long().to(device)
      data = data.view(-1, 784)
    
      data_dist = dist.MultivariateNormal(data, args.sigma*args.sigma*torch.eye(784).to(device))
      data_pert = data_dist.sample().clamp(x_min, x_max).to(device)
    
      output = model(data_pert)
      loss = loss_fn(output, target)
      total_loss += loss.item()

      output_nat = model(data)
      nat_loss = loss_fn(output_nat, target)
      total_nat_loss += nat_loss.item()

      if args.nat:
        nat_loss.backward()
      else:
        loss.backward()
      optimizer.step()   

    total_loss /= len(train_loader.dataset)
    total_nat_loss /= len(train_loader.dataset)
    return total_loss, total_nat_loss

  def test(epoch):
    model.eval()
    total_loss = 0 # SRR loss
    total_nat_loss = 0 # Natural loss
    for _, (data, target) in enumerate(test_loader):
      data, target = data.float().to(device), target.long().to(device)
      data = data.view(-1, 784)
    
      data_dist = dist.MultivariateNormal(data, args.sigma*args.sigma*torch.eye(784).to(device))
      data_pert = data_dist.sample().clamp(x_min, x_max).to(device)
    
      output = model(data_pert)
      loss = loss_fn(output, target)
      total_loss += loss.item()

      output_nat = model(data)
      nat_loss = loss_fn(output_nat, target)
      total_nat_loss += nat_loss.item()
      
    total_loss /= len(test_loader.dataset)
    total_nat_loss /= len(test_loader.dataset)
    return total_loss, total_nat_loss

  # Train the network
  print("Starting network training")
  for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
      train_loss, train_nat_loss  = train(epoch)
      with torch.no_grad():
        test_loss, test_nat_loss = test(epoch)
        
      print("[epoch %03d]  train loss: %.5f, train_nat loss: %.5f, test loss: %.5f, test_nat loss: %.5f" % (epoch, train_loss, train_nat_loss, test_loss, test_nat_loss))

      # Save SRR loss and natural loss in each epoch
      if args.nat:
        torch.save({"train_loss": train_loss, "train_nat_loss": train_nat_loss, "test_loss": test_loss, "test_nat_loss":test_nat_loss}, f'./data/wsummary/mnist_simplemlp_nat_weighted_summary_sigma_{args.sigma}_run_{args.run}_epoch_{epoch}.pth')
      else:
        torch.save({"train_loss": train_loss, "train_nat_loss": train_nat_loss, "test_loss": test_loss, "test_nat_loss":test_nat_loss}, f'./data/wsummary/mnist_simplemlp_stat_weighted_summary_sigma_{args.sigma}_run_{args.run}_epoch_{epoch}.pth')
  print("Training Done.")
  print(f"Final Loss: {test_loss}")

  # Save final model state
  if args.nat:
    torch.save(model.state_dict(), f'./data/mnist_simplemlp_nat_weighted_modeldict_sigma_{args.sigma}_run_{args.run}_epoch_{epoch}.pth')
  else:
    torch.save(model.state_dict(), f'./data/mnist_simplemlp_stat_weighted_modeldict_sigma_{args.sigma}_run_{args.run}_epoch_{epoch}.pth')

if __name__ == '__main__':
  # Read command arguments
  parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
  parser.add_argument('--sigma', default = 0.3, type=float, help='s.d.')
  parser.add_argument('--resume', '-r', action='store_true', help='resume')
  parser.add_argument('--epochs', default=1000, type=int, help='number of training epochs')
  parser.add_argument('--nat', action='store_true', help='natural training')
  parser.add_argument('--run', default=0, type=int, help='run number')
  parser.add_argument('--lr', default= 5e-5, type=float, help='learning rate')
  args = parser.parse_args()
  
  seed = args.run
  np.random.seed(seed)
  torch.manual_seed(seed)
  
  run()
