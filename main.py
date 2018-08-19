import train
import utils

import os
import argparse

import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description='Temporal-Spatial ResNet Video Translater')

# Option
parser.add_argument('--train', '-t', action='store_true',
                                help='Start a new training session.')
parser.add_argument('--resume', '-r', action='store_true',
                                help='Resume training from checkpoint.')
parser.add_argument('--split_video', '-s', action='store_true',
                                help='Split video for training images.')

# Train
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-2, 
                                help='Learning rate of network.')
parser.add_argument('--batch_size', '-batch', type=int, default=5,
                                help='Batch size in training step.')
parser.add_argument('--lr_decay_epoch', '-decay', type=int, default=40,
                                help='Epoch number where learning rate starts to decay.')
parser.add_argument('--log_frequency', '-log', type=int, default=25, 
                                help='Logging frequency in training steps.')
parser.add_argument('--sample_frequency', '-sample', type=int, default=1,
                                help='Sampling frequency in training epochs.')
parser.add_argument('--max_epoch', '-epoch', type=int, default=100, 
                                help='Max epochs in training.')

# Data Location
parser.add_argument('--dataset_a_dir', '-a', type=str, default='image_a', 
                                help='Training-set frames location.')
parser.add_argument('--dataset_b_dir', '-b', type=str, default='image_b', 
                                help='Teacher-set frames location.')
parser.add_argument('--sample_dir', '-test', type=str, default='image_test', 
                                help='Sample-set frames location.')
parser.add_argument('--generate_video', '-g', action='store_true',
                                help='Generate video result in sampling.')
parser.add_argument('--result_dir', '-result', type=str, default='result', 
                                help='Sampling result location.')
parser.add_argument('--num_workers', '-worker', type=int, default=0, 
                                help='The number of dataset workers')

# Video Information
parser.add_argument('--video_a_input', '-va', type=str, default='a.mp4',
                                help='File name of input video for training-set.')
parser.add_argument('--video_b_input', '-vb', type=str, default='b.mp4',
                                help='File name of input video for teacher-set.')
parser.add_argument('--video_output_dir', '-vo', type=str, default='result_video',
                                help='File name of input video for teacher-set.')
parser.add_argument('--video_sample_input', '-vs', type=str, default='sample.mp4', 
                                help='File name of input video for sampling.')
parser.add_argument('--fps', '-fps', type=int, default=10, 
                                help='FPS of input and generated video.')

# Network Type
parser.add_argument('--network', '-net', type=str, default='resnet', 
                                help='Training-set frames location.')

# ResNet Information
parser.add_argument('--image_size', '-img', type=int, default=64,
                                help='The number of downsampling layer in network.')
parser.add_argument('--channel', '-c', type=int, default=16,
                                help='The number of output channels in translators first downsampling layer.')
parser.add_argument('--num_block', '-block', type=int, default=14,
                                help='The number of resnet blocks in network.')
parser.add_argument('--num_downsample', '-downsample', type=int, default=2,
                                help='The number of downsampling layer in network.')

# U-Net Information
parser.add_argument('--u_depth', '-ud', type=int, default=4,
                                help='The depth of U-Net.')
parser.add_argument('--u_channel', '-uc', type=int, default=16,
                                help='The number of channels in the first layer of U-Net.')
                       

# Frame Pool
parser.add_argument('--pool_size', '-pool', type=int, default=9,
                                help='The length of frame sequence as network input.')
parser.add_argument('--train_step', '-t_step', type=int, default=3,
                                help='The step of frame sequence.')
parser.add_argument('--sample_step', '-s_step', type=int, default=3,
                                help='The length of frame sequence as network input.')


config = parser.parse_args()

if config.train or config.resume:
    if config.split_video:
        print('-- Start Splitting Video to Frames')
        utils.split_video(config.video_a_input, config.dataset_a_dir)
        print('-- Complete Splitting %s' % config.video_a_input)
        utils.split_video(config.video_b_input, config.dataset_b_dir)
        print('-- Complete Splitting %s' % config.video_b_input)
        utils.split_video(config.video_sample_input, config.sample_dir)
        print('-- Complete Splitting %s' % config.video_sample_input)
    train.train_network(config)
elif config.generate_video:
    if config.split_video:
        print('-- Start Splitting Video to Frames')
        utils.split_video(config.video_sample_input, config.sample_dir)
        print('-- Complete Splitting %s' % config.video_sample_input)
    train.test_network(config)
    print('-- Start Generating Video from Frames')
    utils.generate_video_from_epoch(config, 0)
    print('-- Result Video Saved')
else:
    print('-- Command Error')
    print('-- Please Enter Add -h Argument to See Usage')
