import torch.distributed as dist

from model import *
from dataset import *

import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from statistics import mean


class Train:
    def __init__(self, args):

        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_l1 = args.wgt_l1
        self.wgt_gan = args.wgt_gan

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        # self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data
        self.direction = args.direction


        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        # if self.gpu_ids and torch.cuda.is_available():
        #     self.device = torch.device("cuda:%d" % self.gpu_ids[0])
        #     torch.cuda.set_device(self.gpu_ids[0])
        # else:
        #     self.device = torch.device("cpu")

    def save(self, dir_chck, netG, netD, optimG, optimD, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        if self.mode == "train_single":
            torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                        'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                       '%s/model_epoch%04d.pth' % (dir_chck, epoch), _use_new_zipfile_serialization=False)

        if self.mode == "train_multi":
            torch.save({'netG': netG.module.state_dict(), 'netD': netD.module.state_dict(),
                        'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                       '%s/model_epoch%04d.pth' % (dir_chck, epoch), _use_new_zipfile_serialization=False)

    def load(self, dir_chck, netG, netD=[], optimG=[], optimD=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        if mode == 'train_single':
            netG.load_state_dict(dict_net['netG'])
            netD.load_state_dict(dict_net['netD'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])

            return netG, netD, optimG, optimD, epoch

        elif mode == 'train_multi':
            netG.module.load_state_dict(dict_net['netG'])
            netD.module.load_state_dict(dict_net['netD'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])

            return netG, netD, optimG, optimD, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])

            return netG, epoch

    def preprocess(self, data):
        normalize = Normalize()
        randflip = RandomFlip()
        rescale = Rescale((self.ny_load, self.nx_load))
        randomcrop = RandomCrop((self.ny_out, self.nx_out))
        totensor = ToTensor()
        return totensor(randomcrop(rescale(randflip(normalize(data)))))

    def deprocess(self, data):
        tonumpy = ToNumpy()
        denomalize = Denomalize()
        return denomalize(tonumpy(data))

    def train(self, gpu, ngpus_per_node, args):

        mode = self.mode
        device = self.device
        gpu_ids = gpu   # fix origin code
        torch.cuda.set_device(gpu_ids)

        if args.mode =='train_multi':
            ngpus_per_node = torch.cuda.device_count()
            args.world_size = ngpus_per_node * args.world_size

            print("Use GPU: {} for training".format(gpu_ids))

            args.rank = args.rank * ngpus_per_node + gpu

            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_l1 = self.wgt_l1
        wgt_gan = self.wgt_gan

        # batch_size = self.batch_size
        batch_size = int(self.batch_size / ngpus_per_node)
        num_workers = int(self.num_workers / ngpus_per_node)

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data, 'train')
        dir_data_val = os.path.join(self.dir_data, name_data, 'val')

        dir_log_train = os.path.join(self.dir_log, self.scope, name_data, 'train')
        dir_log_val = os.path.join(self.dir_log, self.scope, name_data, 'val')

        transform_train = transforms.Compose([Normalize(), RandomFlip(), Rescale((self.ny_load, self.nx_load)), RandomCrop((self.ny_in, self.nx_in)), ToTensor()])
        transform_val = transforms.Compose([Normalize(), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_train = Dataset(dir_data_train, direction=self.direction, data_type=self.data_type, nch=self.nch_in, transform=transform_train)
        dataset_val = Dataset(dir_data_val, direction=self.direction, data_type=self.data_type, nch=self.nch_in, transform=transform_val)
        if mode == 'train_single':
            loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8,drop_last=True)
            loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8,drop_last=True)

        elif mode == 'train_multi':
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
            loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=num_workers, sampler=train_sampler)

            train_sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val)
            loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=(train_sampler_val is None), num_workers=num_workers, sampler=train_sampler_val)

        num_train = len(dataset_train)
        num_val = len(dataset_val)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
        num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))

        ## setup network
        netG = UNet(nch_in, nch_out, nch_ker, norm).cuda(gpu_ids)
        netD = Discriminator(2*nch_in, nch_ker, norm).cuda(gpu_ids)

        ## multi-gpu
        if mode == 'train_multi':
            netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[gpu_ids], find_unused_parameters=True)
            netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[gpu_ids], find_unused_parameters=True)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_L1 = nn.L1Loss().to(device) # L1
        fn_GAN = nn.BCEWithLogitsLoss().to(device)

        paramsG = netG.parameters()
        paramsD = netD.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(paramsD, lr=lr_D, betas=(self.beta1, 0.999))

        # schedG = get_scheduler(optimG, self.opts)
        # schedD = get_scheduler(optimD, self.opts)

        # schedG = torch.optim.lr_scheduler.ExponentialLR(optimG, gamma=0.9)
        # schedD = torch.optim.lr_scheduler.ExponentialLR(optimD, gamma=0.9)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG, netD, optimG, optimD, st_epoch = self.load(dir_chck, netG, netD, optimG, optimD, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)
        writer_val = SummaryWriter(log_dir=dir_log_val)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()
            netD.train()

            loss_G_l1_train = []
            loss_G_gan_train = []
            loss_D_real_train = []
            loss_D_fake_train = []

            for i, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)
                input = data['dataA'].to(device)
                label = data['dataB'].to(device)

                # forward netG
                output = netG(input)

                # backward netD
                fake = torch.cat([input, output], dim=1)
                real = torch.cat([input, label], dim=1)

                set_requires_grad(netD, True)
                optimD.zero_grad()

                pred_real = netD(real)
                pred_fake = netD(fake.detach())

                loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimD.step()

                # backward netG
                fake = torch.cat([input, output], dim=1)

                set_requires_grad(netD, False)
                optimG.zero_grad()

                pred_fake = netD(fake)

                loss_G_gan = fn_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_l1 = fn_L1(output, label)
                loss_G = (wgt_l1 * loss_G_l1) + (wgt_gan * loss_G_gan)

                loss_G.backward()
                optimG.step()


                # get losses
                loss_G_l1_train += [loss_G_l1.item()]
                loss_G_gan_train += [loss_G_gan.item()]
                loss_D_fake_train += [loss_D_fake.item()]
                loss_D_real_train += [loss_D_real.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                      'GEN L1: %.4f GEN GAN: %.4f DISC FAKE: %.4f DISC REAL: %.4f'
                      % (epoch, i, num_batch_train,
                         mean(loss_G_l1_train), mean(loss_G_gan_train), mean(loss_D_fake_train), mean(loss_D_real_train)))

                if should(num_freq_disp):
                    ## show output
                    input = transform_inv(input)
                    output = transform_inv(output)
                    label = transform_inv(label)

                    writer_train.add_images('input', input, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('output', output, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('label', label, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

                    ## show predict
                    pred_fake = transform_inv(pred_fake)
                    pred_real = transform_inv(pred_real)

                    writer_train.add_images('pred_fake', pred_fake, num_batch_train * (epoch - 1) + i, dataformats='NHWC')
                    writer_train.add_images('pred_real', pred_real, num_batch_train * (epoch - 1) + i, dataformats='NHWC')

            writer_train.add_scalar('loss_G_l1', mean(loss_G_l1_train), epoch)
            writer_train.add_scalar('loss_G_gan', mean(loss_G_gan_train), epoch)
            writer_train.add_scalar('loss_D_fake', mean(loss_D_fake_train), epoch)
            writer_train.add_scalar('loss_D_real', mean(loss_D_real_train), epoch)

            ## validation phase
            with torch.no_grad():
                netG.eval()
                netD.eval()
                # netG.train()
                # netD.train()

                loss_G_l1_val = []
                loss_G_gan_val = []
                loss_D_real_val = []
                loss_D_fake_val = []

                for i, data in enumerate(loader_val, 1):
                    def should(freq):
                        return freq > 0 and (i % freq == 0 or i == num_batch_val)

                    input = data['dataA'].to(device)
                    label = data['dataB'].to(device)

                    # forward netG
                    output = netG(input)

                    fake = torch.cat([input, output], dim=1)
                    real = torch.cat([input, label], dim=1)

                    # forward netD
                    pred_fake = netD(fake)
                    pred_real = netD(real)

                    loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                    loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    loss_G_gan = fn_GAN(pred_fake, torch.ones_like(pred_fake))
                    loss_G_l1 = fn_L1(output, label)
                    loss_G = (wgt_l1 * loss_G_l1) + (wgt_gan * loss_G_gan)

                    loss_G_l1_val += [loss_G_l1.item()]
                    loss_G_gan_val += [loss_G_gan.item()]
                    loss_D_real_val += [loss_D_real.item()]
                    loss_D_fake_val += [loss_D_fake.item()]

                    print('VALID: EPOCH %d: BATCH %04d/%04d: '
                          'GEN L1: %.4f GEN GAN: %.4f DISC FAKE: %.4f DISC REAL: %.4f'
                          % (epoch, i, num_batch_val,
                             mean(loss_G_l1_val), mean(loss_G_gan_val), mean(loss_D_fake_val), mean(loss_D_real_val)))

                    if should(num_freq_disp):
                        ## show output
                        input = transform_inv(input)
                        output = transform_inv(output)
                        label = transform_inv(label)

                        writer_val.add_images('input', input, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        writer_val.add_images('output', output, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        writer_val.add_images('label', label, num_batch_val * (epoch - 1) + i, dataformats='NHWC')

                        ## show predict
                        pred_fake = transform_inv(pred_fake)
                        pred_real = transform_inv(pred_real)

                        writer_val.add_images('pred_fake', pred_fake, num_batch_val * (epoch - 1) + i, dataformats='NHWC')
                        writer_val.add_images('pred_real', pred_real, num_batch_val * (epoch - 1) + i, dataformats='NHWC')

                writer_val.add_scalar('loss_G_l1', mean(loss_G_l1_val), epoch)
                writer_val.add_scalar('loss_G_gan', mean(loss_G_gan_val), epoch)
                writer_val.add_scalar('loss_D_fake', mean(loss_D_fake_val), epoch)
                writer_val.add_scalar('loss_D_real', mean(loss_D_real_val), epoch)

            # update schduler
            # schedG.step()
            # schedD.step()

            ## save
#if (epoch % num_freq_save) == 0:
#               if args.rank == 0:
#                   self.save(dir_chck, netG, netD, optimG, optimD, epoch)

        writer_train.close()
        writer_val.close()

    def test(self):
        mode = self.mode

        batch_size = self.batch_size
        device = self.device
        #gpu_ids = self.gpu_ids
        gpu_ids = 0 

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        norm = self.norm

        name_data = self.name_data

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_result = os.path.join(self.dir_result, self.scope, name_data)
        dir_result_save = os.path.join(dir_result, 'images')
        if not os.path.exists(dir_result_save):
            os.makedirs(dir_result_save)

        dir_data_test = os.path.join(self.dir_data, name_data, 'test')

        transform_test = transforms.Compose([Normalize(), ToTensor()])
        transform_inv = transforms.Compose([ToNumpy(), Denomalize()])

        dataset_test = Dataset(dir_data_test, direction=self.direction, data_type=self.data_type, nch=self.nch_in, transform=transform_test)

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        num_test = len(dataset_test)

        num_batch_test = int((num_test / batch_size) + ((num_test % batch_size) != 0))

        ## setup network
        #netG = UNet(nch_in, nch_out, nch_ker, norm)
        netG = UNet(nch_in, nch_out, nch_ker, norm).cuda(gpu_ids)
        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_L1 = nn.L1Loss().to(device)  # L1

        ## load from checkpoints
        st_epoch = 0

        netG, st_epoch = self.load(dir_chck, netG, mode=mode)

        ## test phase
        with torch.no_grad():
            netG.eval()
            # netG.train()

            loss_G_l1_test = []

            for i, data in enumerate(loader_test, 1):
                input = data['dataA'].to(device)
                label = data['dataB'].to(device)

                output = netG(input)

                loss_G_l1 = fn_L1(output, label)

                loss_G_l1_test += [loss_G_l1.item()]

                input = transform_inv(input)
                output = transform_inv(output)
                label = transform_inv(label)

                for j in range(label.shape[0]):
                    name = batch_size * (i - 1) + j
                    fileset = {'name': name,
                                'input': "%04d-input.png" % name,
                                'output': "%04d-output.png" % name,
                                'label': "%04d-label.png" % name}

                    plt.imsave(os.path.join(dir_result_save, fileset['input']), input[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['output']), output[j, :, :, :].squeeze())
                    plt.imsave(os.path.join(dir_result_save, fileset['label']), label[j, :, :, :].squeeze())

                    append_index(dir_result, fileset)

                print('TEST: %d/%d: LOSS: %.6f' % (i, num_batch_test, loss_G_l1.item()))
            print('TEST: AVERAGE LOSS: %.6f' % (mean(loss_G_l1_test)))


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
