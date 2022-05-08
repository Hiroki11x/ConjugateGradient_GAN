import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def build_dataset(opt):
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = datasets.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imagesize),
                                    transforms.CenterCrop(opt.imagesize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        nc=3
    elif opt.dataset == 'lsun':

        '''
        1st pip install lmdb
        2ns git clone https://github.com/fyu/lsun
        3rd python3 download.py -c bedroom -o /groups1/gcb50275/lsun     
        '''

        classes = [ c + '_train' for c in opt.classes.split(',')]
        dataset = datasets.LSUN(root=opt.dataroot, classes=classes,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imagesize),
                                transforms.CenterCrop(opt.imagesize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3
    elif opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imagesize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3

    elif opt.dataset == 'celeba':
        dataset = datasets.ImageFolder(root=opt.dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imagesize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        nc=3

    elif opt.dataset == 'mnist':
        dataset = datasets.MNIST(root=opt.dataroot, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imagesize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))
        nc=1

    elif opt.dataset == 'fake':
        dataset = datasets.FakeData(image_size=(3, opt.imagesize, opt.imagesize),
                                transform=transforms.ToTensor())
        nc=3

    assert dataset
    return dataset, nc
