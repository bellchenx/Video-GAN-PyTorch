import os
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_loader(config, data_dir):
    root = os.path.join(os.path.abspath(os.curdir), data_dir)
    dataset = ImageFolder(
        root=root,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers
    )
    return loader

def norm(image):
    out = (image - 0.5)/0.5
    return out

def denorm(image):
    out = image * 0.5 + 0.5
    return out.clamp(0, 1)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('-- Total Number of Parameters: %d' % num_params)
    print('-- Network Architecture')
    print(net)


class Image_Pool(object):
    def __init__(self, config, step_num, batch_size=5):
        self.max_num = 0
        self.items = []
        self.length = config.pool_size
        self.step_num = step_num
        self.batch_size = batch_size
        self.start_idx = 0
        self.image_size = config.image_size
        if self.length < self.step_num:
            self.step_num = self.length

    def __call__(self):
        out = []

        if self.batch_size == 1 and self.check():
            for image_idx in range(self.start_idx, self.start_idx + self.length):
                out.append(self.items[image_idx])
            out = torch.stack(out, dim=2)
            self.start_idx += self.step_num            
            return out.view(1, 3, self.length, self.image_size, -1)

        for batch_idx in range(0, self.batch_size):
            if self.check():
                batch = []
                for image_idx in range(self.start_idx, self.start_idx + self.length):
                    batch.append(self.items[image_idx])
                batch = torch.stack(batch, dim=2)
                out.append(batch)
                self.start_idx += self.step_num
        out = torch.cat(out, dim=0)
        return out

    def add(self, new):
        self.items.append(new)
        self.max_num += 1

    def reset(self):
        self.start_idx = 0

    def check(self):
        if self.start_idx + self.length < self.max_num:
            return True
        else:
            return False
           

def split_video(video_name, output_dir):
    output_dir = os.path.join(output_dir, 'frames')
    if os.path.isdir(output_dir):
        os.mkdir(output_dir)
    command = 'ffmpeg -loglevel quiet -i %s %s' % (
        video_name, os.path.join(output_dir, 'Frame-%%05d.jpg'))
    os.system(command)


def generate_video_from_epoch(config, epoch):
    image = 'Epoch-%d-%%05d.jpg' % epoch
    image = os.path.join(config.result_dir, image)
    video = 'Epoch-%03d.mp4' % epoch
    video = os.path.join(config.video_output_dir, video)
    command = 'ffmpeg -loglevel quiet -framerate %d -i %s -vcodec mpeg4 %s' % (
        config.fps, image, video)
    os.system(command)
    if config.train or config.resume:
        command = 'rm -rf %s/*' %(config.result_dir)
        os.system(command)
