#! /usr/bin/env python3

from __future__ import division

import argparse
import tqdm
import numpy as np

from terminaltables import AsciiTable

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info
from pytorchyolo.utils.datasets_dcc import ListDataset
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from models.dcc2023_base import DCC2023Model
import math
from pytorch_msssim import ms_ssim

def compute_bpp(out_net):
    # size = out_net['x_hat'].size()
    # num_pixels = size[0] * size[2] * size[3]
    num_pixels=out_net['s_hat'].size(0)*512*512

    base_bpp= sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["base_likelihoods"].values()
    )
    # enhance_bpp = sum(
    #     (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
    #     for likelihoods in out_net["enhance_likelihoods"].values()
    # )
    # total_bpp = (base_bpp + enhance_bpp).item()
    return base_bpp  #,total_bpp
    # return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    #            for likelihoods in out_net['likelihoods'].values()).item()

def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    # return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.).item())
    return ms_ssim(a, b, data_range=1.)

def evaluate_model_file(model_path, weights_path, img_path, class_names, batch_size=8, img_size=416,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    dataloader = _create_validation_data_loader(
        img_path, batch_size, img_size, n_cpu)

    model = DCC2023Model()
    model_dict=model.state_dict()

    checkpoint = torch.load(weights_path, map_location='cuda')
    print(f'test_epoch is {checkpoint["epoch"]}')
    # model.load_state_dict(checkpoint["state_dict"])
    pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to('cuda')
    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    count = 0
    # total_bpp=0
    base_bpp=0
    # ssim_t=0
    # psnr_t=0
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for _, imgs,imgs_lr, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        imgs_lr = Variable(imgs_lr.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs_net = model(imgs,imgs_lr)
            outputs=outputs_net['t_hat']
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)
            count+=1
            base_bpp_tmp=compute_bpp(outputs_net)
            base_bpp+=base_bpp_tmp

            # ssim_t+=compute_msssim(imgs, outputs_net["x_hat"])
            # psnr_t += compute_psnr(imgs, outputs_net["x_hat"])

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    # PSNR = psnr_t / count
    # MS_SSIM = ssim_t / count
    base_bpp = base_bpp / count
    # total_bpp = total_bpp / count
    # print(f'average_PSNR: {PSNR:.2f}dB')
    # print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    # print(f'average_Bit-rate: {total_bpp:.3f} bpp')
    print(f'average_base_bpp: {base_bpp:.3f} ')

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output


def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="/home/adminroot/taofei/YOLO/Pytorch-YOLOv3/config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="/home/adminroot/taofei/DCC2023fuxian/base_coco_fixbn/0.0483/checkpoint_best.pth.tar", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="/home/adminroot/taofei/YOLO/Pytorch-YOLOv3/config/coco_dcc.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=512, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Load configuration from data file
    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # List of class names

    precision, recall, AP, f1, ap_class = evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        class_names,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True)


if __name__ == "__main__":
    run()
