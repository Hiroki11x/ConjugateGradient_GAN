import datetime

def date_str():
    dt_now = datetime.datetime.now()
    date_str = str(dt_now.year) + "_" + str(dt_now.month) + "_" + str(dt_now.day) +\
    "_" + str(dt_now.hour) + "_" + str(dt_now.minute)
    return date_str

def set_otf(opt, wandb_project_name):

    """
    return : e.g. const_lr_cifar10_RESNET_cgd_fr/bs64-betas0.5-0.999-eps1e-12-lr0.01-clip-0.01-2022_2_1_7_1
    """

    date_suffix = date_str()
    if opt.SN == True:
        outf = wandb_project_name + '/' + 'bs{}'.format(opt.batchsize) + 'betas{}-{}'.format(opt.beta1, opt.beta2) + '-eps{}'.format(opt.eps) \
                + '-lrD{}-lrG{}'.format(opt.lr_D, opt.lr_G) + '-clip-{}'.format(opt.clip_value) + '-seeds{}'.format(opt.manualSeed) + "-" + date_suffix
    else:
        outf = wandb_project_name + '/' + 'bs{}'.format(opt.batchsize) + 'betas{}-{}'.format(opt.beta1, opt.beta2) + '-eps{}'.format(opt.eps) \
                + '-lrD{}-lrG{}'.format(opt.lr, opt.lr_G) + '-clip-{}'.format(opt.clip_value) + '-seeds{}'.format(opt.manualSeed) + "-" + date_suffix
    return outf
