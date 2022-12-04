# Conjugate Gradient Method for Generative Adversarial Networks

## Abstract
Generative models such as used for image generation are required to solve the Jensen–Shannon divergence minimization problem between the model distribution and the data distribution, which is computationally infeasible.
Generative Adversarial Networks (GANs)  formulate this problem as a game between two models, a generator and a discriminator, whose learning can be formulated in the context of game theory and the local Nash equilibrium (LNE).
This optimization is more complicated than minimizing a single objective function. Hence, it would be difficult to show stability and optimality for the existing methods for this optimization. 
Here, we propose applying the conjugate gradient method that can solve stably and quickly general large-scale stationary point problems to the LNE problem in GANs.
We give proof and convergence analysis under mild assumptions showing that the proposed method converges to a LNE with three different learning rate update rules, including a constant learning rate as the first attempt ever. 
Finally, we present results that the proposed method outperforms stochastic gradient descent (SGD), momentum SGD, and achieves competitive FID score with Adam in terms of FID score.

## Additional Experimental Results of Rebuttal

We additionally conducted experiments on SNGAN w/ ResNet generator as diminishing return experiments. We report the best FID and the best-10 FID by using grid search to find hyperparameters.
It should be noted that the previous study [Miy+2017] used the Chainer framework, while our implementation uses Pytorch.
In our experiments, Adam updated FID scores of [Miy+2017] because of sufficient hyperparameter search.
However, Adam has stronger hyperparameter sensitivities, and Conjugate gradient methods outperform the other optimizers in the average of the Best-10 FIDs.


|                               | Adam         | SGD        | Momentum SGD  | CGD_DY      | CGD_FR     | CGD_FR_PRP | CGD_HS     | CGD_HS_DY  | CGD_HZ     | CGD_PRP    |
|-------------------------------|--------------|------------|---------------|-------------|------------|------------|------------|------------|------------|------------|
| Miy+2017 (Constant LR)        | 21.7         | -          | -             | -           | -          | -          | -          | -          | -          | -          |
| Ours (Constant LR)            | 19.38        | 35.50      | 35.42         | 30.76       | 26.03      | 34.47      | 32.92      | 34.49      | 32.78      | 31.51      |
| Ours (Diminishing LR)         | 75.86        | 43.67      | 76.07         | 38.11       | 39.27      | (WIP)      | (WIP)      | (WIP)      | (WIP)      | (WIP)      |
| Ours (Constant LR) / Best-10    | 51.16±33.71  | 41.09±8.13 | 82.15±51.82   | 33.70±2.04  | 29.63±2.26 | 34.78±2.17 | 34.82±1.07 | 34.12±1.31 | 34.28±0.96 | 34.53±2.01 |
| Ours (Diminishing LR) / Best-10 | 135.86±32.65 | 51.80±7.84 | 210.10±59.333 | 53.11±12.89 | 47.56±8.82 | (WIP)      | (WIP)      | (WIP)      | (WIP)      | (WIP)      |


![](figs/best-10-avg_fid_sngan_resnet_constant.png)
![](figs/best-10-avg_fid_sngan_resnet_diminishing.png)

## Prerequisites

```sh
gcc==7.4.0
python >= 3.7
cuda == 11.1
cudnn == 8.1
```

## Downloads
- [MNIST Datasets](http://yann.lecun.com/exdb/mnist/)
- [CIFAR10 Datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CelebA Datasets](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Pretrained Inception Model](https://github.com/mseitzer/pytorch-fid/releases) for Calculationg FID


## Installation

#### Install Dependent Libraries

```sh
pip install -r requirements.txt
```

#### Fix Environment Path

```sh
vim ./exp/env_common.sh
```

#### Fix Wandb Entity Path

For example, if you want to do a MNIST on DCGAN w/ SN grid search for the ConstantLR case, you will need to modify the following file.

```sh
vim ./sweep_config/ConstantLR_DCGANSN_MNIST/sgd.yaml
```

Please change entity name `XXXXXX` to your wandb entitiy.


```yaml
project: ConstantLR_mnist_DCGANSN_sgd
entity: XXXXXX
program: main.py
method: grid
```

## Sweep

This section shows how to grid-search sgd's hyperparameters. Other optimizers can be executed in the same way.

#### ConstantLR DCGAN w/SN MNIST

```sh
cd exp/sweep_scripts/ConstantLR_DCGANSN_MNIST/
./sweep_agent_sgd.sh
```

#### ConstantLR DCGAN w/ SN CIFAR10

```sh
cd exp/sweep_scripts/ConstantLR_DCGANSN_CIFAR10/
./sweep_agent_sgd.sh
```

#### DiminishingLR DCGAN w/ SN MNIST

```sh
cd exp/sweep_scripts/InvSqrtLR_DCGANSN_MNIST/
./sweep_agent_sgd.sh
```

#### DiminishingLR DCGAN w/ SN CIFAR10

```sh
cd exp/sweep_scripts/InvSqrtLR_DCGANSN_CIFAR10/
./sweep_agent_sgd.sh
```


#### DiminishingLR DCGAN w/ SN CIFAR10

```sh
cd exp/sweep_scripts/DiminishingLR_DCGANSN_CIFAR10/
./sweep_agent_sgd.sh
```
