import argparse

## dist
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import torch.backends.cudnn as cudnn
from train import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

## setup parse
parser = argparse.ArgumentParser(description='Train the pix2pix network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')

parser.add_argument('--mode', default='train_single', choices=['train_single', 'train_multi', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--scope', default='pix2pix', dest='scope')
parser.add_argument('--norm', type=str, default='inorm', dest='norm')
parser.add_argument('--name_data', type=str, default='res', dest='name_data')

parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./log', dest='dir_log')

parser.add_argument('--dir_data', default='./datasets', dest='dir_data')
parser.add_argument('--dir_result', default='./results', dest='dir_result')

parser.add_argument('--num_epoch', type=int,  default=300, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=16, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=2e-4, dest='lr_G')
parser.add_argument('--lr_D', type=float, default=2e-4, dest='lr_D')

parser.add_argument('--num_freq_disp', type=int,  default=10, dest='num_freq_disp')
# parser.add_argument('--num_freq_save', type=int,  default=10, dest='num_freq_save')
parser.add_argument('--num_freq_save', type=int,  default=1, dest='num_freq_save')
##################################################### 위에꺼 수정할 것

parser.add_argument('--lr_policy', type=str, default='linear', choices=['linear', 'step', 'plateau', 'cosine'], dest='lr_policy')
parser.add_argument('--n_epochs', type=int, default=100, dest='n_epochs')
parser.add_argument('--n_epochs_decay', type=int, default=100, dest='n_epochs_decay')
parser.add_argument('--lr_decay_iters', type=int, default=50, dest='lr_decay_iters')

parser.add_argument('--wgt_l1', type=float, default=1e2, dest='wgt_l1')
parser.add_argument('--wgt_gan', type=float, default=1e0, dest='wgt_gan')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')
parser.add_argument('--beta1', default=0.5, dest='beta1')

parser.add_argument('--ny_in', type=int, default=256, dest='ny_in')
parser.add_argument('--nx_in', type=int, default=256, dest='nx_in')
parser.add_argument('--nch_in', type=int, default=3, dest='nch_in')

parser.add_argument('--ny_load', type=int, default=286, dest='ny_load')
parser.add_argument('--nx_load', type=int, default=286, dest='nx_load')
parser.add_argument('--nch_load', type=int, default=3, dest='nch_load')

parser.add_argument('--ny_out', type=int, default=256, dest='ny_out')
parser.add_argument('--nx_out', type=int, default=256, dest='nx_out')
parser.add_argument('--nch_out', type=int, default=3, dest='nch_out')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')

parser.add_argument('--data_type', default='float32', dest='data_type')
parser.add_argument('--direction', default='A2B', dest='direction')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train_single':
        TRAINER.train(ARGS.gpu,1,ARGS)
    elif ARGS.mode == 'train_multi':
        gpu_devices = ','.join([str(id) for id in ARGS.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        ngpus_per_node = torch.cuda.device_count()
        mp.spawn(TRAINER.train, nprocs=ngpus_per_node, args=(ngpus_per_node,ARGS))
    elif ARGS.mode == 'test':
        TRAINER.test()

if __name__ == '__main__':
    main()
