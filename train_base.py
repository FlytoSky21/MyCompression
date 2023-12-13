# -*- coding:utf-8 -*-
# @Time: 2023/11/16 21:02
# @Author: TaoFei
# @FileName: train.py.py
# @Software: PyCharm

import argparse
import math
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.vimeo90k_dataset import Vimeo90KDataset
# from datasets.ImageNet300k import Vimeo90KDataset

from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models.dcc2023_base import DCC2023Model
from torch.utils.tensorboard import SummaryWriter
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    print("Length of union_params:", len(union_params))
    print("Length of params_dict.keys():", len(params_dict.keys()))

    # 找到以 "yolov3" 开头的参数
    yolov3_params = {
        n: p
        for n, p in params_dict.items()
        if n.startswith('yolov3')
    }

    # 输出参数数量
    print("Number of yolov3 parameters:", len(yolov3_params))

    assert len(inter_params) == 0
    assert len(union_params) == len(params_dict.keys()) - len(yolov3_params)

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["base_bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["base_likelihoods"].values()
        )
        # out["enhance_bpp_loss"] = sum(
        #     (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        #     for likelihoods in output["enhance_likelihoods"].values()
        # )
        # out["bpp_loss"] = out["base_bpp_loss"] + out["enhance_bpp_loss"]
        if self.type == 'mse':
            # out["mse_loss"] = self.mse(output["x_hat"], target)
            out["smse_loss"] = self.mse(output["s_hat"], output["s"])
            out["loss"] = self.lmbda * 255 ** 2 * (0.006 * out["smse_loss"]) + out["base_bpp_loss"]
            # out["loss"] = self.lmbda * 255 ** 2 * (out["mse_loss"] + 0.006 * out["smse_loss"]) + out["bpp_loss"]
        else:
            out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
            out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, writer, type='mse'
):
    model.train()
    model.apply(fix_bn)
    device = next(model.parameters()).device
    loss = AverageMeter()
    # iterations = len(train_dataloader)

    for i, d in enumerate(train_dataloader):
        img = d[0].to(device)
        img_lr = d[1].to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(img, img_lr)

        out_criterion = criterion(out_net, img)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        loss.update(out_criterion["loss"])

        if i % 1000 == 0:
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i * len(img)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tsMSE loss: {out_criterion["smse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["base_bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
                # average_loss = total_loss / i
                # global_step = (epoch - 1) * iterations + i
                # writer.add_scalar("Training Loss", average_loss, global_step)
                # 每隔一定步数记录一下损失
                writer.add_scalar("train/loss", out_criterion["loss"], epoch * len(train_dataloader) + i)
                # writer.add_scalar("train/mse_loss", out_criterion["mse_loss"], epoch * len(train_dataloader) + i)
                writer.add_scalar("train/smse_loss", out_criterion["smse_loss"],
                                  epoch * len(train_dataloader) + i)
                writer.add_scalar("train/base_bpp_loss", out_criterion["base_bpp_loss"],
                                  epoch * len(train_dataloader) + i)
                writer.add_scalar("train/aux_loss", aux_loss, epoch * len(train_dataloader) + i)
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i * len(img)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMS_SSIM loss: {out_criterion["ms_ssim_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["base_bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
    # 记录平均总体训练损失
    writer.add_scalar("train/avg_loss", loss.avg, epoch)


def test_epoch(epoch, test_dataloader, model, criterion, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    if type == 'mse':
        loss = AverageMeter()
        base_bpp_loss = AverageMeter()
        smse_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                img = d[0].to(device)
                img_lr = d[1].to(device)
                # d = d.to(device)
                out_net = model(img, img_lr)
                out_criterion = criterion(out_net, img)

                aux_loss.update(model.aux_loss())
                base_bpp_loss.update(out_criterion["base_bpp_loss"])
                loss.update(out_criterion["loss"])
                smse_loss.update(out_criterion["smse_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tsMSE loss: {smse_loss.avg:.3f} |"
            f"\tBpp loss: {base_bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    else:
        loss = AverageMeter()
        bpp_loss = AverageMeter()
        ms_ssim_loss = AverageMeter()
        aux_loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.3f} |"
            f"\tMS_SSIM loss: {ms_ssim_loss.avg:.3f} |"
            f"\tBpp loss: {bpp_loss.avg:.2f} |"
            f"\tAux loss: {aux_loss.avg:.2f}\n"
        )

    return loss.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=300,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.0483,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=128,
    )
    parser.add_argument(
        "--lr_epoch", nargs='+', type=int
    )
    parser.add_argument(
        "--continue_train", action="store_true", default=True
    )
    args = parser.parse_args(argv)
    return args


def save_checkpoint(state, is_best, epoch, save_path):
    torch.save(state, save_path + "/checkpoint_latest.pth.tar")
    if epoch % 5 == 0:
        torch.save(state, save_path + "/checkpoint_" + str(epoch) + ".pth.tar")
    if is_best:
        torch.save(state, save_path + "/checkpoint_best.pth.tar")


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    save_path = os.path.join(args.save_path, str(args.lmbda))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "tensorboard/")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    writer = SummaryWriter(save_path + "tensorboard/")

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size)]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size)]
    )

    train_dataset = Vimeo90KDataset(args.dataset, split="train", transform=train_transforms)
    test_dataset = Vimeo90KDataset(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f'train device is  {device}')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    net = DCC2023Model()
    # print(net)
    # for layer in net.children():
    #     print((layer))
    net = net.to(device)

    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # milestones = args.lr_epoch
    # print("milestones: ", milestones)
    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    # 创建一个学习了调度器，该调度器根据优化过程中监测的某个指标的变化来自动调整学习率。
    # 在这里，指定的指标是最小化的，当这个指标不再减小时，学习率会被调整。
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=10)
    criterion = RateDistortionLoss(lmbda=args.lmbda, type=type)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            writer,
            type
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, type)
        writer.add_scalar('test_loss', loss, epoch)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                save_path,
            )
        # 如果学习率已经降低了4次，就停止训练
        if optimizer.param_groups[0]['lr'] <= 1e-4 * (0.1 ** 4):
            break
    writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
