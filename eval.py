# -*- coding:utf-8 -*-
# @Time: 2023/11/23 10:30
# @Author: TaoFei
# @FileName: eval.py
# @Software: PyCharm

import torch
import torch.nn.functional as F
from torchvision import transforms
from models.dcc2023 import DCC2023Model
import warnings
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image

warnings.filterwarnings("ignore")

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    # return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.).item())
    return ms_ssim(a, b, data_range=1.)


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]

    base_bpp= sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["base_likelihoods"].values())
    enhance_bpp = sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["enhance_likelihoods"].values())
    total_bpp = (base_bpp + enhance_bpp).item()
    return total_bpp
    # return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    #            for likelihoods in out_net['likelihoods'].values()).item()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", default=True,help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", default='/home/adminroot/taofei/DCC2023fuxian/fuwuqi/0_0483/checkpoint_best.pth.tar',type=str, help="Path to a checkpoint")
    parser.add_argument("--data", default='/home/adminroot/taofei/dataset/Kodak24', type=str, help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=True
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def main(argv):
    # 验证重建性能
    args = parse_args(argv)
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'

    net = DCC2023Model()
    net = net.to(device)

    net.eval()
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    dictory = {}

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # for k, v in checkpoint["state_dict"].items():
        #     dictory[k.replace("module.", "")] = v
        # net.load_state_dict(dictory)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_dict}

        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        # net.load_state_dict(checkpoint["state_dict"])
        print(f'test-epoch is : {checkpoint["epoch"]}')


    save_path='./rec_images'
    for img_name in img_list:
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        img_lr = img.resize((w // 8, h // 8), Image.BICUBIC)
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        x_lr = transforms.ToTensor()(img_lr).unsqueeze(0).to(device)
        count += 1
        with torch.no_grad():
            s = time.time()
            out_net = net.forward(x, x_lr)
            e = time.time()
            total_time += (e - s)
            out_net['x_hat'].clamp_(0, 1)
            reconstructed_img = transforms.ToPILImage()(out_net['x_hat'].squeeze(0).cpu())
            # 注意：.squeeze(0) 用于移除 batch 维度，.cpu() 用于将数据从 GPU 移回 CPU

            # 保存图像为 PNG 文件
            rec_path=os.path.join(save_path,img_name)
            reconstructed_img.save(rec_path)
            psnr_img = compute_psnr(x, out_net["x_hat"])
            ms_ssim_img = compute_msssim(x, out_net["x_hat"])
            bpp_img = compute_bpp(out_net)
            # print(f'PSNR: {psnr_img:.2f}dB')
            # print(f'MS-SSIM: {ms_ssim_img:.2f}dB')
            # print(f'Bit-rate: {bpp_img:.3f}bpp')
            PSNR += psnr_img
            MS_SSIM += ms_ssim_img
            Bit_rate += bpp_img
    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    total_time = total_time / count
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.3f} ms')


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
