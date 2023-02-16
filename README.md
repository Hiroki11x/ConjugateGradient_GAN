# Conjugate Gradient Method for Generative Adversarial Networks @AISTATS2023

## Abstract
Generative models such as used for image generation are required to solve the Jensen–Shannon divergence minimization problem between the model distribution and the data distribution, which is computationally infeasible.
Generative Adversarial Networks (GANs)  formulate this problem as a game between two models, a generator and a discriminator, whose learning can be formulated in the context of game theory and the local Nash equilibrium (LNE).
This optimization is more complicated than minimizing a single objective function. Hence, it would be difficult to show stability and optimality for the existing methods for this optimization. 
Here, we propose applying the conjugate gradient method that can solve stably and quickly general large-scale stationary point problems to the LNE problem in GANs.
We give proof and convergence analysis under mild assumptions showing that the proposed method converges to a LNE with three different learning rate update rules, including a constant learning rate as the first attempt ever. 
Finally, we present results that the proposed method outperforms stochastic gradient descent (SGD), momentum SGD, and achieves competitive FID score with Adam in terms of FID score.


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

For example, if you want to do a CIFAR10 on SNGAN w/ ResNet Generator, grid search for the ConstantLR case, you will need to modify the following file.

```sh
vim ./sweep_config/CL_RESNET_CIFAR10/sgd.yaml
```

Please change entity name `XXXXXX` to your wandb entitiy.


```yaml
project: CL_RESNET_CIFAR10
entity: XXXXXX
program: main.py
method: grid
```

## Sweep

This section shows how to grid-search sgd's hyperparameters. Other optimizers can be executed in the same way.


#### ConstantLR SNGAN w/ ResNet Generator on CIFAR10

```sh
cd exp/sweep_scripts/CL_RESNET_CIFAR10/
./sweep_agent_sgd.sh
```


#### DiminishingLR SNGAN w/ ResNet Generator on CIFAR10

```sh
cd exp/sweep_scripts/DL_RESNET_CIFAR10/
./sweep_agent_sgd.sh
```


