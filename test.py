import os
import argparse
from FDEGCNet import *
from torchvision import transforms

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import logging
import lpips  # Ensure the LPIPS library is imported

import pyiqa
from datasets import Datasets, TestKodakDataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter
from PIL import Image, ImageFile

torch.backends.cudnn.enabled = True

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None  
ImageFile.LOAD_TRUNCATED_IMAGES = True  

# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 8192
print_freq = 100
cal_step = 40
warmup_step = 0#  // gpu_num
batch_size = 4
tot_epoch = 1000000
tot_step = 2500000
decay_interval = 2200000
lr_decay = 0.1
image_size = 256
logger = logging.getLogger("ImageCompression")
tb_logger = None
global_step = 0
save_model_freq = 50000
test_step = 10000
out_channel_N = 192
out_channel_M = 320
parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

parser.add_argument('-n', '--name', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('-t', '--test', default='',
        help='test dataset')
parser.add_argument('--config', dest='config', required=False,
        help = 'hyperparameter in json format')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--val', dest='val_path', required=True, help='the path of validation dataset')

def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, batch_size, print_freq, \
        out_channel_M, out_channel_N, save_model_freq, test_step
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        if train_lambda < 4096:
            out_channel_N = 128
            out_channel_M = 192
        else:
            out_channel_N = 192
            out_channel_M = 320
    if 'batch_size' in config:
        batch_size = config['batch_size']
    if "print_freq" in config:
        print_freq = config['print_freq']
    if "test_step" in config:
        test_step = config['test_step']
    if "save_model_freq" in config:
        save_model_freq = config['save_model_freq']
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']
    if 'out_channel_N' in config:
        out_channel_N = config['out_channel_N']
    if 'out_channel_M' in config:
        out_channel_M = config['out_channel_M']


def test(step):
    lpips_model = lpips.LPIPS(net='alex').to(device)  # Initialize the LPIPS model
    vifp_metric = pyiqa.create_metric('vif').to(device)  # Initialize the VIFp metric

    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        sumLpips = 0
        sum_VIFp = 0
        cnt = 0

        for batch_idx, input in enumerate(test_loader):
            clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)

            # Ensure correct range and type
            clipped_recon_image = clipped_recon_image.clamp(0.0, 1.0).float().to(device)
            input = input.clamp(0.0, 1.0).float().to(device)
            mse_loss, bpp_feature, bpp_z, bpp = (
                torch.mean(mse_loss), 
                torch.mean(bpp_feature), 
                torch.mean(bpp_z), 
                torch.mean(bpp)
            )
            psnr = 10 * (torch.log10(1. / mse_loss))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)  # MS-SSIM calculation moved to CPU
            msssimDB = -10 * (torch.log10(1 - msssim))
            sumMsssimDB += msssimDB
            sumMsssim += msssim

            # Calculate LPIPS value
            lpips_value = lpips_model(clipped_recon_image, input).item()
            sumLpips += lpips_value

            # Calculate VIFp and catch exceptions
            try:
                vifp_score = vifp_metric(clipped_recon_image, input).item()
                sum_VIFp += vifp_score
            except Exception as e:
                logger.error(f"Error calculating VIFp: {e}")

            cnt += 1
            logger.info(f"Num: {cnt}, Bpp: {bpp:.6f}, PSNR: {psnr:.6f}, MS-SSIM: {msssim:.6f}, "
                        f"MS-SSIM-DB: {msssimDB:.6f}, LPIPS: {lpips_value:.6f}, VIFp: {vifp_score:.6f}")

        # Calculate averages
        avg_lpips = sumLpips / cnt
        avg_vifp = sum_VIFp / cnt
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt

        logger.info(f"Dataset Average result---Dataset Num: {cnt}, Bpp: {sumBpp:.6f}, PSNR: {sumPsnr:.6f}, "
                    f"MS-SSIM: {sumMsssim:.6f}, MS-SSIM-DB: {sumMsssimDB:.6f}, Avg LPIPS: {avg_lpips:.6f}, "
                    f"VIFp: {avg_vifp:.6f}")
         # Save the log to file
        with open("log.log", "a") as log_file:
            log_file.write(f"Test on Kodak dataset: model-{step}, train_lambda-{train_lambda}\n")
            log_file.write(f"Dataset Average result---Dataset Num: {cnt}, Bpp: {sumBpp:.6f}, PSNR: {sumPsnr:.6f}, "
                           f"MS-SSIM: {sumMsssim:.6f}, MS-SSIM-DB: {sumMsssimDB:.6f}, Avg LPIPS: {avg_lpips:.6f}, "
                           f"VIFp: {avg_vifp:.6f}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    tb_logger = None
    logger.setLevel(logging.INFO)
    logger.info("image compression test")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)
    logger.info("out_channel_N:{}, out_channel_M:{}".format(out_channel_N, out_channel_M))
    model = ImageCompressor(out_channel_N)
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = model.to(device)
    net = torch.nn.DataParallel(net, device_ids=[1])
    global test_loader
 
    test_dataset = TestKodakDataset(data_dir=args.val_path)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True)
    test(global_step)
