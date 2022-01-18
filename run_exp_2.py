from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import numpy as np

from exp_6_2.eval import eval_metric

model_epsilons = [0.0, 0.157, 0.5]
model_adv_epsilons = [0.157]
eval_epsilons = [0.0, 0.157, 0.5]
eval_adv_epsilons = [0.157]
model_epsilons_str = ', '.join(map(str, model_epsilons))
eval_epsilons_str = ', '.join(map(str, eval_epsilons))
model_adv_epsilons_str = ', '.join(map(str, model_adv_epsilons))
eval_adv_epsilons_str = ', '.join(map(str, eval_adv_epsilons))

runs = 5

# Train network on CIFAR-10 for natural training, two versions of corruption training, and PGD adversarial training.
# Progressively smaller learning rates are used over training
print('Beginning training of Wide ResNet networks on CIFAR-10')
for run in range(runs):
    print("Training run #", run)
    for train_epsilon in model_epsilons:
        print("Corruption training epsilon: ", train_epsilon)
        cmd0 = 'python -m exp_6_2.train --epsilon={} --epochs=20 --lr=0.01 --run={}'.format(train_epsilon, run)
        cmd1 = 'python -m exp_6_2.train --resume --epsilon={} --epochs=5 --lr=0.002 --run={}'.format(train_epsilon, run)
        cmd2 = 'python -m exp_6_2.train --resume --epsilon={} --epochs=5 --lr=0.0004 --run={}'.format(train_epsilon, run)
        os.system(cmd0)
        os.system(cmd1)
        os.system(cmd2)
    for train_adv_epsilon in model_adv_epsilons:
        print("PGD training epsilon: ", train_adv_epsilon)
        cmd0 = 'python -m exp_6_2.train --adv --epsilon={} --epochs=20 --lr=0.01 --run={}'.format(train_adv_epsilon, run)
        cmd1 = 'python -m exp_6_2.train --adv --resume --epsilon={} --epochs=5 --lr=0.002 --run={}'.format(train_adv_epsilon, run)
        cmd2 = 'python -m exp_6_2.train --adv --resume --epsilon={} --epochs=5 --lr=0.0004 --run={}'.format(train_adv_epsilon, run)
        os.system(cmd0)
        os.system(cmd1)
        os.system(cmd2)
    

# Calculate metrics (A-TSRM and adv. accuracy), evaluating each trained network on each metric
print('Beginning metric evaluation')
# Evaluation on train/test set respectively
avg_train_metrics = np.empty([len(eval_epsilons) + len(eval_adv_epsilons), len(model_epsilons) + len(model_adv_epsilons)])
avg_test_metrics = np.empty([len(eval_epsilons) + len(eval_adv_epsilons), len(model_epsilons) + len(model_adv_epsilons)])
for run in range(runs):
    print("Metric evaluation for training run #", run)
    train_metrics = np.empty([len(eval_epsilons) + len(eval_adv_epsilons), len(model_epsilons) + len(model_adv_epsilons)])
    test_metrics = np.empty([len(eval_epsilons) + len(eval_adv_epsilons), len(model_epsilons) + len(model_adv_epsilons)])
    
    # Corruption training, A-TRSM evaluation
    for idx, train_epsilon in enumerate(model_epsilons):
        print("Corruption training epsilon: ", train_epsilon, ", Evaluate on A-TSRM epsilons: ", model_epsilons_str)
        filename = './data/ckpt_stat_epsilon_{}_run_{}.pth'.format(train_epsilon, run)
        train_metric_col = eval_metric(filename, eval_epsilons, adv=False, train=True)
        test_metric_col = eval_metric(filename, eval_epsilons, adv=False, train=False)
        train_metrics[:len(eval_epsilons), idx] = np.array(train_metric_col)
        test_metrics[:len(eval_epsilons), idx] = np.array(test_metric_col)
        
    # Corruption training, adversarial evaluation
    for idx, train_epsilon in enumerate(model_epsilons):
        print("Corruption training epsilon: ", train_epsilon, ", Evaluate on adversarial epsilons: ", model_adv_epsilons_str)
        filename = './data/ckpt_stat_epsilon_{}_run_{}.pth'.format(train_epsilon, run)
        train_metric_col = eval_metric(filename, eval_adv_epsilons, adv=True, train=True)
        test_metric_col = eval_metric(filename, eval_adv_epsilons, adv=True, train=False)
        train_metrics[len(eval_epsilons):, idx] = np.array(train_metric_col)
        test_metrics[len(eval_epsilons):, idx] = np.array(test_metric_col)

    # Adversarial training, A-TSRM evaluation
    for idx, train_adv_epsilon in enumerate(model_adv_epsilons):
        print("Adversarial training epsilon: ", train_adv_epsilon, ", Evaluate on A-TSRM epsilons: ", model_epsilons_str)
        filename = './data/ckpt_adv_epsilon_{}_run_{}.pth'.format(train_adv_epsilon, run)
        train_metric_col = eval_metric(filename, eval_epsilons, adv=False, train=True)
        test_metric_col = eval_metric(filename, eval_epsilons, adv=False, train=False)
        train_metrics[:len(eval_epsilons), len(model_epsilons) + idx] = np.array(train_metric_col)
        test_metrics[:len(eval_epsilons), len(model_epsilons) + idx] = np.array(test_metric_col)

    # Adversarial training, adversarial evaluation
    for idx, train_adv_epsilon in enumerate(model_adv_epsilons):
        print("Adversarial training epsilon: ", train_adv_epsilon, ", Evaluate on adversarial epsilons: ", model_adv_epsilons_str)
        filename = './data/ckpt_adv_epsilon_{}_run_{}.pth'.format(train_adv_epsilon, run)
        train_metric_col = eval_metric(filename, eval_adv_epsilons, adv=True, train=True)
        test_metric_col = eval_metric(filename, eval_adv_epsilons, adv=True, train=False)
        train_metrics[len(eval_epsilons):, len(model_epsilons) + idx] = np.array(train_metric_col)
        test_metrics[len(eval_epsilons):, len(model_epsilons) + idx] = np.array(test_metric_col)
    
    avg_train_metrics += train_metrics
    avg_test_metrics += test_metrics

    np.savetxt("./results/cifar10_metrics_train_run_{}.csv".format(run), train_metrics, fmt='%1.1f', delimiter=',', header='Networks trained with' 
              ' corruptions (epsilon = {}) and adversarially (epsilon = {}) along columns, THEN evaluated on training set using A-TRSM (epsilon = {}) '
               ' and adversarial acc (epsilon = {}) along rows'.format(model_epsilons_str, model_adv_epsilons_str, eval_epsilons_str, eval_adv_epsilons_str))

    np.savetxt("./results/cifar10_metrics_test_run_{}.csv".format(run), test_metrics, fmt='%1.1f', delimiter=',', header='Networks trained with' 
              ' corruptions (epsilon = {}) and adversarially (epsilon = {}) along columns, THEN evaluated on test set using A-TRSM (epsilon = {}) '
               ' and adversarial acc (epsilon = {}) along rows'.format(model_epsilons_str, model_adv_epsilons_str, eval_epsilons_str, eval_adv_epsilons_str))

np.savetxt("./results/cifar10_metrics_train_avg.csv".format(run), avg_train_metrics, fmt='%1.1f', delimiter=',', header='Networks trained with' 
            ' corruptions (epsilon = {}) and adversarially (epsilon = {}) along columns, THEN evaluated on training set using A-TRSM (epsilon = {}) '
            ' and adversarial acc (epsilon = {}) along rows'.format(model_epsilons_str, model_adv_epsilons_str, eval_epsilons_str, eval_adv_epsilons_str))

np.savetxt("./results/cifar10_metrics_test_avg.csv".format(run), avg_test_metrics, fmt='%1.1f', delimiter=',', header='Networks trained with' 
            ' corruptions (epsilon = {}) and adversarially (epsilon = {}) along columns, THEN evaluated on test set using A-TRSM (epsilon = {}) '
            ' and adversarial acc (epsilon = {}) along rows'.format(model_epsilons_str, model_adv_epsilons_str, eval_epsilons_str, eval_adv_epsilons_str))
