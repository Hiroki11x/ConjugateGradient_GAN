# Conjugate Gradient Method for Generative Adversarial Networks

## Abstract
While the generative model has many advantages, it is not feasible to calculate the Jensen–Shannon divergence of the density function of the data and the density function of the model of deep neural networks; for this reason, various alternative approaches have been developed. Generative adversarial networks (GANs) can be used to formulate this problem as a discriminative problem with two models, a generator and a discriminator whose learning can be formulated in the context of game theory and the local Nash equilibrium. Since this optimization is more difficult than minimization of a single objective function, we propose to apply the conjugate gradient method to solve the local Nash equilibrium problem in GANs. We give a proof and convergence analysis under mild assumptions showing that the proposed method converges to a local Nash equilibrium with three different learning-rate schedules including a constant learning rate. Furthermore, we demonstrate the convergence of a simple toy problem to a local Nash equilibrium and compare the proposed method with other optimization methods in experiments using real-world data, finding that the proposed method outperforms stochastic gradient descent (SGD) and momentum SGD.

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

```sh
pip install -r requirements.txt
```

## Sweep