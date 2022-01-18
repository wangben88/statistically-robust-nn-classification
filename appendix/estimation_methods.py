import torch
import numpy as np
import scipy.stats as st
import torch.distributions as dist

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def tsrm_carpentier(X_array, y_array, model, epsilon=0.3, lcb=0.4, beta=0.1, batch_size=10, num_batches=100):
    """Algorithm from 'Adaptive Strategy for Stratified Monte Carlo Sampling' (Carpentier et al.)"""
    model.eval()
    
    N = len(y_array)
    sample_sizes = np.zeros(N)
    emp_mean = np.zeros(N)
    emp_variance = np.zeros(N)
    
    precomp_iterations = 2
    correct_arr = np.empty((precomp_iterations, N))
    for i in range(precomp_iterations):
        data_dist = dist.Uniform(low = torch.max(X_array-epsilon, torch.Tensor([0]).to(device)), high = torch.min(X_array+epsilon, torch.Tensor([1]).to(device)))
        data_pert = data_dist.sample().to(device)
        target = y_array
    
        output = model(data_pert)
        _, preds = torch.max(output, dim=1)
        correct = preds == target
        correct_arr[i, :] = correct.cpu().numpy()
        
        emp_mean = (sample_sizes * emp_mean + correct.cpu().numpy())/(sample_sizes + 1)
        sample_sizes += 1
        
    emp_variance = np.var(correct_arr, axis=0, ddof=1)
        
    for i in range(num_batches):
        ## STEP 1 : Compute which points to sample from
        variance_dec = (emp_variance + 2*beta/np.sqrt(sample_sizes))/sample_sizes
        eval_idxs = np.argpartition(variance_dec, -batch_size)[-batch_size:]
        
        ## STEP 2: Sample from these points
        data = X_array[eval_idxs]
        target = y_array[eval_idxs]
        
        data_dist = dist.Uniform(low = torch.max(data-epsilon, torch.Tensor([0]).to(device)), high = torch.min(data+epsilon, torch.Tensor([1]).to(device)))
        data_pert = data_dist.sample().to(device)
        
        output = model(data_pert)
        preds = output.argmax(dim=1)
        
        # STEP 3: Update estimates
        correct = (preds == target)
        emp_mean[eval_idxs] = (sample_sizes[eval_idxs] * emp_mean[eval_idxs] + correct.cpu().numpy())/(sample_sizes[eval_idxs] + 1)
        emp_variance[eval_idxs] = (sample_sizes[eval_idxs] - 1)/(sample_sizes[eval_idxs]) * emp_variance[eval_idxs] + \
                            np.square(correct.cpu().numpy() - emp_mean[eval_idxs])/(sample_sizes[eval_idxs] + 1)
        sample_sizes[eval_idxs] += 1
        
        
    return emp_mean, emp_variance, sample_sizes

def tsrm_mc(X_array, y_array, model, epsilon=0.3, batch_size=10, epochs=10):
    """Simple Monte Carlo estimation of TSRM as detailed in paper"""
    model.eval()
    
    N = len(y_array)
    total_correct = 0
    for epoch in range(epochs):
      for i in range(N//batch_size):
        data = X_array[i*batch_size: (i+1)*batch_size]
        data = data.view(-1, 784)
        target = y_array[i*batch_size: (i+1)*batch_size]
    
        data_dist = dist.Uniform(low=torch.max(data - epsilon, torch.Tensor([0]).to(device)), high=torch.min(data + epsilon, torch.Tensor([1]).to(device)))
      
        data_pert = data_dist.sample().to(device)
    
        output = model(data_pert)
        preds = output.argmax(dim=1)
    
        total_correct += (preds == target).sum().float().item()
 
    return total_correct / (N*epochs)
