# Statistically Robust Neural Network Classification

Code to reproduce the experimental results for [*Statistically Robust Neural Network Classification*](https://proceedings.mlr.press/v161/wang21b.html), UAI 2021.

## Experiment 6.1
To reproduce the results of Experiment 6.1, run the following from the base directory:

`python run_exp_1.py`

This will:
1. Train the NN classifier on MNIST using natural and corrupted training methods, as described in the paper;
2. Evaluate the TSRM metric on each trained NN at a number of epsilon values;
3. Collate the results and produce the plot of Figure 1.

## Experiment 6.2
Likewise, to reproduce the results of Experiment 6.2, run the following:

`python run_exp_2.py`

This will:
1. Train the wide ResNet CNN classifier on CIFAR-10 using natural, corruption and adversarial training methods;
2. Evaluate the trained networks on natural risk, SRR, and adversarial risk, outputting the results to a csv file (corresponding to results in Table 1).

## Experiment 6.3
Likewise, to reproduce the results of Experiment 6.3, run the following:

`python run_exp_3.py`

This will:
1. Train the NN classifier on MNIST using natural and corrupted training methods (2 networks);
2. Keep track of the natural and SRR weighted cross entropy loss during each epoch of training for both networks;
3. Produce the plot of Figure 2.

## Experiment in Appendix A
Likewise, to reproduce the results of the experiment in Appendix A, run the following (AFTER running Experiment 6.1):

`python run_exp_estimation.py`

This will:
1. Load the naturally trained NN classifier on MNIST from Experiment 6.1;
2. Evaluate the TSRM using both adaptive sampling and monte carlo for this network and 100 datapoints from the MNIST test set;
3. Produce the plot of Figure 3.