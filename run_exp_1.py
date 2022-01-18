from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

epochs = 50
epsilons = [0.0, 0.1, 0.3, 0.5, 0.7]

# Train network on MNIST for natural training, and four versions of corruption training
cmd = 'python -m exp_6_1.train --cauchy --epochs={}'.format(epochs)
os.system(cmd)
for epsilon in epsilons:
    cmd = 'python -m exp_6_1.train --epsilon={} --epochs={}'.format(epsilon, epochs)
    os.system(cmd)

# Calculate points on graph
cmd = 'python -m exp_6_1.eval --results_loc="./results" --model_loc="./data" --prefix="mnist_simplemlp_stat_cauchy"'
os.system(cmd)
for epsilon in epsilons:
    cmd = 'python -m exp_6_1.eval --results_loc="./results" --model_loc="./data" --prefix="mnist_simplemlp_stat_{}"'.format(epsilon)
    os.system(cmd)
    
# Produce figure
cmd = 'python -m exp_6_1.plot'
os.system(cmd)
