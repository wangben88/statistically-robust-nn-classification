# Run this file from the main directory as: python -m appendix.compute_estimates

import torch
from appendix.estimation_methods import tsrm_carpentier, tsrm_mc
from exp_6_1.network import SimpleMlp
import numpy as np
import pickle

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributions as dist

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=100, shuffle=False)

X_tarray = torch.zeros(10000, 784).to(device)
y_tarray = torch.zeros(10000, dtype=torch.long).to(device)
for idx, (data, target) in enumerate(test_loader):
    data, target = data.float().to(device), target.long().to(device)
    data = data.view(-1, 784)
    X_tarray[(idx*100):((idx+1)*100),:] = data
    y_tarray[(idx*100):((idx+1)*100)] = target

model = SimpleMlp()
model.load_state_dict(torch.load("data/mnist_simplemlp_stat_0.0.pth"))
model.to(device)

epoch_counts = [10, 30, 100, 300, 1000]
run_count = 100
carpentier_results = np.empty((len(epoch_counts), run_count))
mc_results = np.empty((len(epoch_counts), run_count))

for epoch_idx, epochs in enumerate(epoch_counts):
    for run in range(run_count):
        emp_means, _, _ = tsrm_carpentier(X_tarray[0:100], y_tarray[0:100], model, batch_size=10, num_batches=epochs*10)
        carpentier_results[epoch_idx, run] = np.mean(emp_means)
        
        mc_result = tsrm_mc(X_tarray[0:100], y_tarray[0:100], model, batch_size=10, epochs=epochs)
        mc_results[epoch_idx, run] = mc_result
    print("Done for ", epochs, " epochs...")
    print(np.std(carpentier_results[epoch_idx, :]), np.std(mc_results[epoch_idx, :]))
    print(np.mean(carpentier_results[epoch_idx, :]), np.mean(mc_results[epoch_idx, :]))
    with open(f'./results/estimation_results2.pickle', 'wb') as handle:
      pickle.dump({'epoch_counts': epoch_counts, 'carpentier_results': carpentier_results, 'mc_results': mc_results}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        
print(carpentier_results, mc_results)


