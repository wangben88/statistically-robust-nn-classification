from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import pickle

import numpy as np
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch.distributions as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from exp_6_1.network import SimpleMlp


def tsrm_mc(X_array, y_array, model, count_runs=5, epsilon=0.3, count_particles=10000):
    lg_ps = []
    for idx in range(count_runs):
        print("run no: ", idx)
        
        sample_ids = np.random.choice(X_array.shape[0], count_particles, replace=True)
        X_init = X_array[sample_ids]
        y_init = y_array[sample_ids]
        
        initial_samp = dist.Uniform(low = torch.max(X_init-epsilon, torch.Tensor([0]).to(device)), high = torch.min(X_init+epsilon, torch.Tensor([1]).to(device)))
        X_samp = initial_samp.sample()
        
        def prop(x):
          y = model(x)
          y_ref = torch.gather(y, 1, (y_init*torch.ones(count_particles, dtype=torch.long).to(device)).unsqueeze(1))

          y_diff = y - y_ref
          y_diff.scatter_(1, (y_init*torch.ones(count_particles, dtype=torch.long).to(device)).unsqueeze(1), -np.inf)

          y_diff, _ = y_diff.max(dim=1)
          return y_diff #.max(dim=1)
        
        s_x = prop(X_samp)
        proportion = torch.sum(s_x >= 0).cpu().numpy()/count_particles
        print("A-TRSM: ", proportion)

        lg_p = np.log(proportion)
        lg_ps.append(lg_p)
        
    outputfile = os.path.join(args.results_loc, args.prefix + '_epsilon_{}_count_particles_{}.pickle'.format(epsilon, count_particles))
    with open(outputfile, 'wb') as handle:
        pickle.dump({'lg_ps':lg_ps}, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  # Read command arguments
  parser = argparse.ArgumentParser(description='Computation of TSRM (total statistical robustness metric)')
  parser.add_argument('--model_loc', help='location of saved model')
  parser.add_argument('--results_loc', help='location to save results')
  parser.add_argument('--prefix', help='prefix for saving results')
  args = parser.parse_args()

  # Load model  
  model = SimpleMlp()
  modelfile = os.path.join(args.model_loc, args.prefix+'.pth')
  model.load_state_dict(torch.load(modelfile))
  model.to(device)

  # Load dataloader
  mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                           transforms.ToTensor()
                         ]))
  test_loader = DataLoader(mnist_test, batch_size = 1000, shuffle=False)
                    
  # Read dataset into a single tensor
  X_array = torch.zeros(10000, 784).to(device)
  y_array = torch.zeros(10000, dtype=torch.long).to(device)
  for idx, (data, target) in enumerate(test_loader):
    data, target = data.float().to(device), target.long().to(device)
    data = data.view(-1, 784)
    X_array[(idx*1000):((idx+1)*1000),:] = data
    y_array[(idx*1000):((idx+1)*1000)] = target

  # For a given NN model, calculate the value of the TSRM for various epsilons
  for epsilon in ([0.001, 0.003, 0.01, 0.03, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]):
      print("Epsilon: ", epsilon)
      tsrm_mc(X_array, y_array, model, count_runs=10, epsilon=epsilon, count_particles=100000)
