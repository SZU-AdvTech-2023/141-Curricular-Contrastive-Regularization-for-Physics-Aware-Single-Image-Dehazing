import argparse
import os

import numpy as np
import torch
import torchvision.transforms as tfs
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from metrics import psnr, ssim
from models.C2PNet import C2PNet

# 解析命令行参数
parser = argparse.ArgumentParser()
#参数指定了数据集名称，可以是'indoor'或'outdoor'
parser.add_argument('-d', '--dataset_name', help='name of dataset', choices=['indoor', 'outdoor'],
                    default='indoor')
#参数指定了保存图像的基本目录，默认是'dehaze_images'
parser.add_argument('--save_dir', type=str, default='dehaze_images', help='dehaze images save path')
parser.add_argument('--save', action='store_true', help='save dehaze images')
opt = parser.parse_args()

dataset = opt.dataset_name

# 创建保存图像的目录
if opt.save:
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    output_dir = os.path.join(opt.save_dir, dataset)
    print("pred_dir:", output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

# 设置数据集和模型路径
if dataset == 'indoor':
    haze_dir = 'data/SOTS/indoor/hazy/'
    clear_dir = 'data/SOTS/indoor/clear/'
   # model_dir = './trained_models/its_train_C2PNet_3_19_default_clcr.pk'
    model_dir = 'trained_models/ITS.pkl'
elif dataset == 'outdoor':
    haze_dir = 'data/SOTS/outdoor/hazy/'
    clear_dir = 'data/SOTS/outdoor/clear/'
    model_dir = 'trained_models/OTS.pkl'
   # model_dir = './trained_models/its_train_C2PNet_3_19_default_clcr.pk'

# 初始化模型
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
net = C2PNet(gps=3, blocks=19)
ckp = torch.load(model_dir)
net = net.to(device)
#print(net.device)
net.load_state_dict(ckp['model'])
net.eval()

# 评估指标列表
psnr_list = []
ssim_list = []

# 处理并评估图像
for im in tqdm(os.listdir(haze_dir)):
    haze = Image.open(os.path.join(haze_dir, im)).convert('RGB')
    if dataset == 'indoor' or dataset == 'outdoor':
        clear_im = im.split('_')[0] + '.png'
    else:
        clear_im = im
    clear = Image.open(os.path.join(clear_dir, clear_im)).convert('RGB')
    haze1 = tfs.ToTensor()(haze)[None, ::]
    haze1 = haze1.to(device)
    clear_no = tfs.ToTensor()(clear)[None, ::]
    with torch.no_grad():
        pred = net(haze1)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    pp = psnr(pred.cpu(), clear_no)
    ss = ssim(pred.cpu(), clear_no)
    psnr_list.append(pp)
    ssim_list.append(ss)

    # 保存去雾图像
    if opt.save:
        vutils.save_image(ts, os.path.join(output_dir, im))


# 输出平均评估指标
print(f'Average PSNR is {np.mean(psnr_list)}')
print(f'Average SSIM is {np.mean(ssim_list)}')
