"""
forked from https://github.com/juntang-zhuang/Adabelief-Optimizer/tree/update_0.2.0/PyTorch_Experiments/wgan/main.py
"""

from __future__ import print_function
import argparse
import os
import random
import wandb
import uuid
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import torchvision.utils as vision_util
import torch.nn.functional as F
from tqdm import tqdm

from utils.metric import FloatMetric, TensorMetric
from utils.lib_version import print_libs_version
from utils.fid_score import calculate_fid_given_paths
from optimizers.set_optim import set_optimizers
from utils.set_model import set_models
from utils.log_utils import date_str, set_otf
from utils.data_utils import build_dataset
from utils.lr_scheduler import build_scheduler

parser = argparse.ArgumentParser()

# Problem Setting
parser.add_argument('--dataset', required=False, default='cifar10', choices=[
                    'cifar10', 'mnist', 'lsun', 'celeba'], help='cifar10 | mnist | ( imagenet | folder | lfw | fake : Not Supported)')
parser.add_argument('--classes', default='bedroom',
                    help='comma separated list of classes for the lsun data set')
parser.add_argument('--dataroot', required=False, default='./',help='path to dataset')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--iters_budget', type=int, default=100000, help='number of iter to train form, default is 100K')

# GAN Specific Setting
parser.add_argument('--imagesize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ndf', type=int, default=64, help='dimension of discrim filters in first conv layer.')
parser.add_argument('--ngf', type=int, default=64, help='dimension of gen filters in first conv layer.')
parser.add_argument("--n_critic", type=int, default=1, help="number of training iter for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="(only for WGAN) lower and upper clip value for disc. weights")
parser.add_argument('--model', default='GAN', choices=['GAN', 'WGAN'], help = 'GAN | (WGAN : Not Supported')
parser.add_argument('--SN', action='store_true', help = 'If you want to use SN when using GAN, set a flag.')

# Hardware Setting and for Reproduction
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda', default=True)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')

# For Optimizer and Learning Rate
parser.add_argument('--scheduler_type', default='ConstLR', choices=['ConstLR', 'ExponentialLR', 'SqrtLR', 'TTScaleLR'], help='ConstLR, ExponentialLR, SqrtLR or TTScaleLR')
parser.add_argument('--lr_D', type=float, default=0.0002, help='learning rate for discrim, default=0.0002')
parser.add_argument('--lr_G', type=float, default=0.0002, help='learning rate for gen, default=0.0002')
parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'momentum_sgd', 'sgd', 'cgd_dy', 'cgd_fr', 'cgd_hs', 'cgd_hz', 'cgd_fr_prp', 'cgd_hs_dy', 'cgd_prp'], help='Optimizer')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', default=0.999, type=float, help='Beta2')
parser.add_argument('--eps',default=1e-8, type=float, help='eps')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--beta_momentum_coeff', default=1.0, type=float, help='beta coefficient for conjugate gradient')

# Logger
parser.add_argument("--update_frequency", type=int, default=1000, help="number of iter frequency for logging")
parser.add_argument("--wandb_entity", type=str, default='XXXXXX', help="entitiy of wandb team")
parser.add_argument('--debug_mode', action = 'store_true')

args = parser.parse_args()
print_libs_version()

# ============= Wandb Setup =============

wandb_project_name = f"{args.scheduler_type}_{args.dataset}_{args.model}_{args.optimizer}"
exp_name_suffix = str(uuid.uuid4())
wandb_exp_name = f"{exp_name_suffix}" # example: XX823748291 
wandb.init(config=args,
        project=wandb_project_name,
        name=wandb_exp_name,
        entity=args.wandb_entity)

# update hyperparams for reflecting wandb sweep
opt = wandb.config
print('Updated HyperParams:')
for k, v in sorted(opt.items()):
    print('\t{}: {}'.format(k, v))

# ============= Initialize and Determine Seeds =============
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True


# ============= Log Dirs Setup =============
output_dir = os.environ['OUTPUT_DIR']
date_str = date_str()
opt.outf = set_otf(opt, wandb_project_name)

try:
    os.makedirs(output_dir + wandb_project_name)
except OSError:
    pass

try:
    os.makedirs(output_dir + opt.outf)
except OSError:
    pass

try:
    os.makedirs(output_dir + opt.outf + '/img')
except OSError:
    pass

# ============= Decide which device we want to run on =============
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")
print(f'device : {device}')

# ============= Build Data Loader =============
dataset, n_channel = build_dataset(opt)
dataloader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=opt.batchsize, 
                                        shuffle=True, 
                                        num_workers=int(opt.workers))
                                        
# ============= Weight Initialization =============
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# ============= Build Model and Loss =============
netD, netG = set_models(weights_init, 
                        model=opt.model, 
                        SN=opt.SN, 
                        ngpu=int(opt.ngpu), nz=int(opt.nz), ngf=int(opt.ngf), ndf=int(opt.ndf), nc=n_channel)

criterion = nn.BCELoss()
fixed_noise = torch.randn(opt.batchsize, int(opt.nz), 1, 1).cuda()
real_label = 1
fake_label = 0

# ============= Setup Optimizer and LR Scheduler =============
optimizerD = set_optimizers(opt.optimizer.lower(), 
                                        netD, 
                                        opt.lr_D,
                                        opt.momentum, 
                                        opt.beta1, opt.beta2, opt.eps, # For Adaptive Optimizer
                                        opt.beta_momentum_coeff
                                        )
optimizerG = set_optimizers(opt.optimizer.lower(), 
                                        netG, 
                                        opt.lr_G,
                                        opt.momentum, 
                                        opt.beta1, opt.beta2, opt.eps, # For Adaptive Optimizer
                                        opt.beta_momentum_coeff
                                        )

scheduler_D, scheduler_G = build_scheduler(opt, optimizerD, optimizerG)

# ============= Convert all training data into png format =============
real_folder = output_dir + f'all_real_imgs_{opt.dataset}'
if not os.path.exists(real_folder):
    os.mkdir(real_folder)
    for i in tqdm(range(len(dataset))):
        vision_util.save_image(dataset[i][0], real_folder + '/{}.png'.format(i), normalize=True)

fake_folder = output_dir + opt.outf +'/'+ f'all_fake_imgs_{opt.dataset}'
if not os.path.exists(fake_folder):
    os.mkdir(fake_folder)


# ============= Setup iter and Epochs =============
iters_per_epoch = int(len(dataset) / opt.batchsize) + 1
epochs = int(opt.iters_budget / iters_per_epoch) + 1

print("Iterations Budget:")
print("\tTotal Iterations: {}".format(opt.iters_budget))
print("\tBatch Size : {}".format(opt.batchsize))
print("\tData Size : {}".format(len(dataset)))
print("\tUpdate Iter Interval : {}".format(opt.update_frequency))
print("\tIterations per Epoch : {}".format(iters_per_epoch))
print("\tTotal Epochs : {}".format(epochs))


# ============= Training Loop =============
iter = 0
for epoch in range(epochs):
    print('Epoch {}'.format(epoch))

    losses_D = TensorMetric('losses_D')
    losses_G = TensorMetric('losses_G')

    norms_D = FloatMetric('norms_D')
    norms_G = FloatMetric('norms_G')

    for i, data in enumerate(dataloader, 0):

        iter += 1

        real_imgs = data[0].cuda()
        batch_size = real_imgs.size(0)

        # For GAN
        label_real = torch.ones(batch_size).cuda()
        label_fake = torch.zeros(batch_size).cuda()

        # ============= Training Discriminator =============
        optimizerD.zero_grad()

        # Sample noise as netG input
        z = torch.randn(batch_size, int(opt.nz), 1, 1).cuda()
        # Generate a batch of images
        gen_imgs = netG(z).detach()

        # ============= Compute Adversarial Loss =============
        if opt.model == 'GAN':
            loss_D = 0.5 * (
                F.binary_cross_entropy(netD(real_imgs).squeeze(), label_real) +
                F.binary_cross_entropy(netD(gen_imgs).squeeze(), label_fake))
        elif opt.model == 'WGAN':
            loss_D = -torch.mean(netD(real_imgs)) + torch.mean(netD(gen_imgs))
        
        losses_D.update(loss_D)

        # ============= Compute Gradient and Backprop =============
        loss_D.backward()
        optimizerD.step()

        # Clip weights of netD
        if opt.model == 'WGAN':
            for p in netD.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        # ============= Compute Grad Norm =============
        norm_D = 0
        parameters_D = [p for p in netD.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters_D:
            param_norm = p.grad.detach().data.norm(2)
            norm_D += param_norm.item() ** 2
        norm_D = norm_D ** 0.5
        norms_D.update(norm_D)

        # ============= Training Generator =============
        # Train the netG every n_critic iterations
        if i % opt.n_critic == 0:

            optimizerG.zero_grad()

            # Generate a batch of images
            gen_imgs = netG(z)

            # ============= Compute Adversarial Loss =============
            if opt.model == 'GAN':
                loss_G = F.binary_cross_entropy(netD(gen_imgs).squeeze(), label_real)
            elif opt.model == 'WGAN':
                # Adversarial loss
                loss_G = -torch.mean(netD(gen_imgs))
            losses_G.update(loss_G)

            # ============= Compute Gradient and Backprop =============
            loss_G.backward()
            optimizerG.step()

            # ============= Compute Grad Norm =============
            norm_G = 0
            parameters_G = [p for p in netG.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters_G:
                param_norm = p.grad.detach().data.norm(2)
                norm_G += param_norm.item() ** 2
            norm_G = norm_G ** 0.5
            norms_G.update(norm_G)

        if iter % opt.update_frequency == 0:
            print(f'update and save at iteration: {iter} / epoch: {epoch}')

            vision_util.save_image(real_imgs,
                    '%s/real_samples.png' % (output_dir + opt.outf + '/img'),
                    normalize=True)
            fake = netG(fixed_noise)
            vision_util.save_image(fake.detach(),
                    '%s/fake_samples_iter_%07d.png' % (output_dir + opt.outf + '/img', iter),
                    normalize=True)
                    
            # ============= FID =============
            fid_batch_size = 256
            fake_image_num_sample = 10000
            generation_loop_iter = int(fake_image_num_sample/fid_batch_size)
            # test netG, and calculate FID score
            netG.eval()
            for i in range(generation_loop_iter):
                noise = torch.randn(fid_batch_size, int(opt.nz), 1, 1).cuda()
                fake = netG(noise)
                for j in range(fake.shape[0]):
                    # replace fake images which is reflected current status
                    vision_util.save_image(fake.detach()[j,...], fake_folder + '/{}.png'.format(j + i * fid_batch_size), normalize=True)
            netG.train()

            # calculate FID score
            fid_value = calculate_fid_given_paths([real_folder, fake_folder],
                                                fid_batch_size//2,
                                                cuda=True)
            print('FID: {}'.format(fid_value))

            wandb.log({
                'avg_losses_D' : losses_D.avg.item(),
                'avg_losses_G' : losses_G.avg.item(),
                'avg_norms_D' : norms_D.avg,
                'avg_norms_G' : norms_G.avg,
                'loss_D': loss_D,
                'loss_G': loss_G,
                'norm_D': norm_D,
                'norm_G': norm_G,
                'fid' : fid_value,
                'epoch': epoch,
                'iter': iter,
                'lr_D': scheduler_D.get_last_lr()[0],
                'lr_G': scheduler_G.get_last_lr()[0],
            })

            print(f'clear accumulated gradients and losses')
            losses_D = TensorMetric('losses_D')
            losses_G = TensorMetric('losses_G')
            norms_D = FloatMetric('norms_D')
            norms_G = FloatMetric('norms_G')

        # ============= Update Scheduler for Each Step =============
        scheduler_G.step()
        scheduler_D.step()
