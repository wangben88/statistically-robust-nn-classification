# Run this file from the main directory as: python -m appendix.plot

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import pickle

import numpy as np
import matplotlib
matplotlib.use('pdf')
#matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

plt.style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
#matplotlib.rc('font', family='Latin Modern Roman')
#mathtext.fontset
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
colors = [None for _ in range(6)]
colors[0] = viridis[0]
colors[1] = plt.get_cmap('Paired').colors[3]
colors[2] = tab20b[13]
colors[3] = (0.3, 0.3, 0.3)
colors[4] = "yellowgreen"
colors[5] = "purple"


def cm2inch(value):
  return value/2.54
    
def plot_new_results():
  """
  Produce Figure 1
  """

  # Load previously collated results
  results = pickle.load(open('./results/estimation_results2.pickle', 'rb'))
  epoch_counts = results['epoch_counts']
  carpentier_results = results['carpentier_results']
  mc_results = results['mc_results']
    
  run_count = carpentier_results.shape[1]

  xs = np.log10(np.array(epoch_counts))

  # Calculate means and std for every total epochs and carpentier/mc
  mean_carpentier = np.mean(carpentier_results, axis=1)
  std_carpentier = np.std(carpentier_results, axis=1)
  mean_mc = np.mean(mc_results, axis=1)
  std_mc = np.std(mc_results, axis=1)

  # Create figure and axes
  fig = plt.figure(figsize=(cm2inch(16.0), cm2inch(12.0)))
  ax = fig.add_subplot(1, 1, 1, xlabel=r'$\log_{10}($epochs$)$', ylabel='Standard deviation of estimate', ylim=(0, 0.01))
  #ax.hlines(y=0, linestyle='--', xmin=xs[0], xmax=xs[-1], linewidth=0.75, color='black') #color='red',
  #ax.hlines(y=-100, color='black', xmin=xs[0], xmax=xs[-1], linestyle='--', linewidth=0.75)

  for idx in range(len(epoch_counts)):
    x = mean_carpentier[idx]
    std = std_carpentier[idx]

    lower = x - std
    upper = min(1.0, x + std)
        

  label = 'adaptive stratified'
  ax.plot(xs, std_carpentier, color=colors[0], linestyle='-', linewidth=1.5, marker='.', label=label)
                                                
  for idx in range(len(epoch_counts)):
    x = mean_mc[idx]
    std = std_mc[idx]

    lower = x - std
    upper = min(1.0, x + std)
        
  label = 'monte carlo'
  ax.plot(xs, std_mc, color=colors[1], linestyle='-', linewidth=1.5, marker='.', label=label)
                                                
                                                

  ax.legend()
  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()

  #fig.savefig(f'./results/mnist_all.svg', bbox_inches='tight')
  fig.savefig(f'./results/estimation.pdf', bbox_inches='tight')
  plt.close(fig)

if __name__ == '__main__':
  plot_new_results()
