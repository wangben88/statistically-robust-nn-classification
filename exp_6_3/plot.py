# Run this file from the main directory as e.g.: python -m exp_6_3.plot --nat --run=0 --sigma=0.3 --epochs=1000

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import torch
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('pdf')
#matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

SMALL_SIZE = 15
MEDIUM_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

viridis = plt.get_cmap('viridis').colors
viridis = [viridis[i] for i in [100, 240, 150, 0]]
tab20b = plt.get_cmap('tab20b').colors
colors = [None for _ in range(5)]
colors[0] = viridis[0]
colors[1] = plt.get_cmap('Paired').colors[3]
colors[2] = tab20b[13]
colors[3] = (0.3, 0.3, 0.3)
colors[4] = "yellowgreen"

parser = argparse.ArgumentParser(description='PyTorch MNIST plots')
parser.add_argument('--sigma', default=0.3, type=float, help="training Gaussian s.d.")
parser.add_argument('--epochs', default=1000, type=int, help="number of training epochs")
parser.add_argument('--run', default=0, type=int, help='training run number')
parser.add_argument('--nat', action='store_true', help='natural training')
args = parser.parse_args()

if args.nat:
  train_str = "nat"
else:
  train_str = "stat"

def cm2inch(value):
  return value/2.54

def extract_weighted_results():
  """
  Collate results into single pickle
  """
  # SRR loss
  train_loss = np.empty(args.epochs)
  test_loss = np.empty(args.epochs)
  # Natural loss
  train_nat_loss = np.empty(args.epochs)
  test_nat_loss = np.empty(args.epochs)
  for epoch in range(1, args.epochs + 1):
    resultfile = './data/wsummary/mnist_simplemlp_{}_weighted_summary_sigma_{}_run_{}_epoch_{}.pth'.format(train_str, args.sigma, args.run, epoch)
    result = torch.load(open(resultfile, 'rb'))
    train_loss[epoch - 1] = result['train_loss']
    test_loss[epoch - 1] = result['test_loss']
    train_nat_loss[epoch - 1] = result['train_nat_loss']
    test_nat_loss[epoch - 1] = result['test_nat_loss']
    
  with open('./results/extracted_weighted_results_{}_{}_{}.pickle'.format(train_str, args.sigma, args.run), 'wb') as handle:
    pickle.dump({'train_loss':train_loss, 'test_loss':test_loss, 'train_nat_loss':train_nat_loss, 'test_nat_loss':test_nat_loss}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def plot_weighted_results(eval_str):
  """
  Use pickle from extracted_weighted_results to plot loss over epochs
  """
  # Load previously collated results
  results = pickle.load(open('./results/extracted_weighted_results_{}_{}_{}.pickle'.format(train_str, args.sigma, args.run), 'rb'))

  if (eval_str == "nat"):
    train_loss = results['train_nat_loss']
    test_loss = results['test_nat_loss']
  else:
    train_loss = results['train_loss']
    test_loss = results['test_loss']
    
  xs = np.arange(args.epochs) + 1

  # Create figure and axes
  fig = plt.figure(figsize=(cm2inch(16.0), cm2inch(12.0)))
  ax = fig.add_subplot(1, 1, 1, xlabel='Epoch number', ylabel='Loss', ylim=(0, 0.005))

  ax.plot(xs, train_loss, color=colors[0], linestyle='-', linewidth=1.5, marker='.', label='Train')

  ax.plot(xs, test_loss, color=colors[1], linestyle='-', linewidth=1.5, marker='.', label='Test')

  ax.legend()
  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()
    
  #fig.savefig(f'./results/mnist_all.svg', bbox_inches='tight')
  fig.savefig('./results/mnist_weighted_{}_{}.pdf'.format(train_str, eval_str), bbox_inches='tight')
  plt.close(fig)

if __name__ == '__main__':
  # Collate metric files
  extract_weighted_results()

  # Produce Figure 1
  plot_weighted_results(eval_str="nat")
  plot_weighted_results(eval_str="stat")
