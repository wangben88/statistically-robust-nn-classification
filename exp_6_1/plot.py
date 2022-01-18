# Run this file from the main directory as: python -m exp_6_1.plot

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

# Plotting parameters
model_epsilons = ["cauchy", 0.0, 0.1, 0.3, 0.5, 0.7]
eval_epsilons = [0.001, 0.003, 0.01, 0.03, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
count_particles = 100000

def cm2inch(value):
  return value/2.54

def extract_new_results():
  """
  Collate results into single pickle
  """
  lg_ps = {}
  for eval_epsilon in eval_epsilons:
    for model_epsilon in model_epsilons:
      resultfile = './results/mnist_simplemlp_stat_{}_epsilon_{}_count_particles_{}.pickle'.format(model_epsilon, eval_epsilon, count_particles)
      result = pickle.load(open(resultfile, 'rb'))
      lg_ps.setdefault(model_epsilon, [])
      lg_ps[model_epsilon].append(result['lg_ps'])
    
  with open(f'./results/extracted_new_results.pickle', 'wb') as handle:
    pickle.dump({'lg_ps':lg_ps, 'eval_epsilons': eval_epsilons, 'model_epsilons': model_epsilons}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def plot_new_results():
  """
  Produce Figure 1
  """

  # Load previously collated results
  results = pickle.load(open('./results/extracted_new_results.pickle', 'rb'))
  lg_ps = results['lg_ps']
  eval_epsilons = results['eval_epsilons']
  model_epsilons = results['model_epsilons']

  xs = np.log10(np.array(eval_epsilons))

  # Calculate means and std for every model for every TSRM metric
  means_ps = {}
  stds_ps = {}
  for model_epsilon in model_epsilons:
    count_points = len(lg_ps[model_epsilon])
    means_ps[model_epsilon] = np.zeros((count_points))
    stds_ps[model_epsilon] = np.zeros((count_points))
    for idx in range(count_points):
      p = np.exp(lg_ps[model_epsilon][idx])
      p = p[p > 0]
      means_ps[model_epsilon][idx] = np.mean(p)
      stds_ps[model_epsilon][idx] = 2*np.std(p)/math.sqrt(p.shape[0])

  # Create figure and axes
  fig = plt.figure(figsize=(cm2inch(16.0), cm2inch(12.0)))
  ax = fig.add_subplot(1, 1, 1, xlabel=r'$\log_{10}(\epsilon)$', ylabel=r'$\log_{10}(I_{total})$', ylim=(-2.0, 0))
  #ax.hlines(y=0, linestyle='--', xmin=xs[0], xmax=xs[-1], linewidth=0.75, color='black') #color='red',
  #ax.hlines(y=-100, color='black', xmin=xs[0], xmax=xs[-1], linestyle='--', linewidth=0.75)

  for jdx, model_epsilon in enumerate(model_epsilons):
    # Add error bars to plot for a given model
    for idx in range(means_ps[model_epsilon].shape[0]):
        x = means_ps[model_epsilon][idx]
        std = stds_ps[model_epsilon][idx]

        lower = x - std
        upper = min(1.0, x + std)
        
        #print(lower, upper)
        #print(np.log(np.array([lower, upper]))/math.log(10))

        if lower < 0:
          ax.scatter(np.array([xs[idx]]), np.log(np.array([x]))/math.log(10), color=colors[jdx], marker='.', linewidth=4)
        else:
          ax.plot(np.array([xs[idx], xs[idx]]), np.log(np.array([lower, upper]))/math.log(10), color=colors[jdx], linewidth=1)

    # Add mean line to plot for a given model
    label = 'natural' if model_epsilon == 0. else r"$\epsilon = {}$".format(model_epsilon)
    ax.plot(xs, np.log10(means_ps[model_epsilon]), color=colors[jdx], linestyle='-', linewidth=1.5, marker='.', label=label)

  ax.legend()
  ax.xaxis.set_tick_params(width=0.5)
  ax.yaxis.set_tick_params(width=0.5)
  ax.spines['left'].set_linewidth(0.5)
  ax.spines['bottom'].set_linewidth(0.5)
  sns.despine()

  #fig.savefig(f'./results/mnist_all.svg', bbox_inches='tight')
  fig.savefig(f'./results/mnist_all.pdf', bbox_inches='tight')
  plt.close(fig)

if __name__ == '__main__':
  # Collate metric files
  extract_new_results()

  # Produce Figure 1
  plot_new_results()
