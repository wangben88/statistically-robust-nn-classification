from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

epochs = 1000
lr = 5e-5
sigma = 0.3

# Train network on MNIST using natural training and corruption training using Gaussian noise
cmd_nat = 'python -m exp_6_3.train --nat --sigma={} --epochs={} --lr={}'.format(sigma, epochs, lr)
cmd_stat = 'python -m exp_6_3.train --sigma={} --epochs={} --lr={}'.format(sigma, epochs, lr)
os.system(cmd_nat)
os.system(cmd_stat)

# Produce plots of Experiment 6.3
cmd_plot_nat = 'python -m exp_6_3.plot --nat --sigma={} --epochs={}'.format(sigma, epochs)
cmd_plot_stat = 'python -m exp_6_3.plot --sigma={} --epochs={}'.format(sigma, epochs)
os.system(cmd_plot_nat)
os.system(cmd_plot_stat)
