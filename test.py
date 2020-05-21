import ntpath
import os
import sys
import warnings

import requests
import numpy as np
import tqdm
from torch import nn
from torchvision.transforms import ToTensor, ToPILImage

from configs import decode_config
from data import create_dataloader
from metric import get_mAP, get_fid
from metric.inception import InceptionV3
from metric.fid_score import calculate_fid_given_paths
from metric.mAP_score import DRNSeg
from models import create_model
from options.test_options import TestOptions
from utils import html, util

from PIL import Image
import requests
from io import BytesIO

def save_images(webpage, visuals, image_path, opt):
    def convert_visuals_to_numpy(visuals):
        for key, t in visuals.items():
            tile = opt.batch_size > 8
            if key == 'labels':
                t = util.tensor2label(t, opt.input_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    visuals = convert_visuals_to_numpy(visuals)

    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims = []
    txts = []
    links = []

    for label, image_numpy in visuals.items():
        image_name = os.path.join(label, '%s.png' % (name))
        save_path = os.path.join(image_dir, image_name)
        util.save_image(image_numpy, save_path, create_dir=True)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=opt.display_winsize)


def check(opt):
    assert opt.serial_batches
    assert opt.no_flip
    assert opt.load_size == opt.crop_size
    assert opt.preprocess == 'resize_and_crop'
    assert opt.batch_size == 1

    if not opt.no_fid:
        assert opt.real_stat_path is not None
    if opt.phase == 'train':
        warnings.warn('You are using training set for inference.')


def calculate_fid(datasetA_path, datasetB_path):
    return calculate_fid_given_paths((datasetA_path, datasetB_path), 1, False, 2048, True)


def get_tensor_from_url(A_url):
    try:
        res = requests.get(A_url)
        A_img = Image.open(BytesIO(res.content)).convert('RGB')
        A = ToTensor()(A_img).unsqueeze(0)
        return {'A': A, 'A_url': A_url, 'A_paths': [A_url]}
    except Exception as why:
        return None


def init_inference_opt():
    """
    /Users/eric/byted/bin/python test.py --dataroot database/horse2zebra/valA \
        --dataset_mode url \
        --results_dir results-pretrained/cycle_gan/horse2zebra/compressed \
        --config_str 16_16_32_16_32_32_16_16 \
        --restore_G_path pretrained/cycle_gan/horse2zebra/compressed/latest_net_G.pth \
        --need_profile \
        --no_fid \
        --real_stat_path real_stat/horse2zebra_B.npz \
        --max_dataset_size -1 \
    """
    opt = TestOptions().parse()
    opt.dataroot = 'database/horse2zebra/valA'
    opt.dataset_mode = 'single'
    opt.results_dir = 'results-pretrained/cycle_gan/horse2zebra/compressed'
    opt.config_str = '16_16_32_16_32_32_16_16'
    opt.restore_G_path = 'pretrained/cycle_gan/horse2zebra/compressed/latest_net_G.pth'
    opt.need_profile = True
    opt.no_fid = True
    opt.real_stat_path = 'real_stat/horse2zebra_B.npz'
    opt.max_dataset_size = -1
    return opt


opt = init_inference_opt()
model = create_model(opt)
model.setup(opt)
config = decode_config(opt.config_str)


def inference(url):
    """return PIL.Image"""
    tensor = get_tensor_from_url(url)
    if not tensor:
        return
    # 在这里对 tensor 进行测试并保存
    url = tensor.get('A_url')
    print(url)
    model.set_input(tensor)  # unpack data from data loader
    model.profile(config)
    model.test(config)  # run inference
    visuals = model.get_current_visuals()  # get image results
    generated = visuals['fake_B'].cpu().squeeze()
    img = ToPILImage()(generated)
    return img


if __name__ == '__main__':
    urls = ['https://storage.googleapis.com/ylq_server/test.jpg']
    for url in urls:
        img = inference(url)

