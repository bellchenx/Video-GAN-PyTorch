import utils
import model

import os
import time
import math

import torch
import torchvision
import tensorboardX

use_cuda = torch.cuda.is_available()
writer = tensorboardX.SummaryWriter()

def variable(var, cuda=None):
    if use_cuda and cuda == None:
        return torch.autograd.Variable(var).cuda()
    if cuda == True:
        return torch.autograd.Variable(var).cuda()
    if cuda == False:
        return torch.autograd.Variable(var)


def adjust_learning_rate(config, optimizer, epoch):
    lr_now = config.learning_rate
    x = epoch / config.lr_decay_epoch - 1
    if epoch > config.lr_decay_epoch:
        lr_now = lr_now / (2**x)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
    return lr_now


def load_network(config, resume=True):
    global net
    global net_epoch
    global time_used
    if config.network == 'resnet':
        net = model.ResNet(config)
    elif config.network == 'resnet_v2':
        net = model.ResNet_v2(config)
    elif config.network == 'unet':
        net = model.UNet(config)
    net_epoch = 0
    time_used = 0
    if resume:
        print('-- Loading Parameters')
        assert os.path.isdir(
            'checkpoint'), '-- Error: No Checkpoint Directory Found!'
        checkpoint = torch.load('./checkpoint/network.nn')
        net = checkpoint['net']
        net_epoch = int(checkpoint['epoch'])
        time_used = float(checkpoint['time'])
    if use_cuda:
        net = net.cuda()
    utils.print_network(net)


def train_network(config):
    load_network(config, config.resume and not config.train)
    net.train()

    loader_a = utils.get_loader(config, config.dataset_a_dir)
    loader_b = utils.get_loader(config, config.dataset_b_dir)

    pool_a = utils.Image_Pool(config, config.train_step, batch_size=config.batch_size)
    pool_b = utils.Image_Pool(config, config.train_step, batch_size=config.batch_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=0)

    print('-- Loading Images')
    for i, (image_a, image_b) in enumerate(zip(loader_a, loader_b)):
        pool_a.add(image_a[0])
        pool_b.add(image_b[0])

    print('-- Start Training from Epoch %d' % (net_epoch + 1))
    total_time = 0
    total_step = 0
    epoch_time = time.time()

    for epoch in range(net_epoch + 1, config.max_epoch + 1):
        lr = adjust_learning_rate(config, optimizer, epoch)
        print('-- Start Training Epoch %d Learning Rate %.4f' % (epoch, lr))
        step_time = time.time()
        pool_a.reset()
        pool_b.reset()
        idx = 0
        while True:
            net.zero_grad()
            result_a = net(variable(pool_a()))
            target = variable(pool_b())
            loss = criterion(result_a, target)
            writer.add_scalar('data/scalar1', loss, total_step)
            loss.backward()
            optimizer.step()

            if not pool_a.check() or not pool_b.check():
                break

            if idx % config.log_frequency == 0 and idx > 0:
                speed = (time.time() - step_time) / config.log_frequency / config.batch_size
                step_time = time.time()
                format_str = 'Training Network: Step %d Batch-Loss: %.4f Speed %.2f sec/batch'
                print(format_str % (idx, loss, speed))
            
            idx += 1
            total_step += 1

        total_time += (time.time() - epoch_time) / 3600
        time_est = (time.time() - epoch_time) * (config.max_epoch - epoch) / 3600
        format_str='-- Epoch %d Completed: Epoch Time: %.2f min Total Time %.2f hours Est Time: %.2f hours'
        print(format_str % (epoch, (time.time() - epoch_time)/60, total_time + time_used, time_est))
        step_time = time.time()
        epoch_time = time.time()

        print('-- Saving Parameters')
        state={'net': net, 'time': time_used + total_time, 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/network.nn')

        if epoch % config.sample_frequency == 0:
            generate(config, epoch)
            if config.generate_video:
                utils.generate_video_from_epoch(config, epoch)



def test_network(config):
    load_network(config, resume=True)
    generate(config, 0)


def generate(config, epoch):
    step_time = time.time()
    loader = utils.get_loader(config, config.sample_dir)
    sample_time = time.time()
    pool = utils.Image_Pool(config, config.sample_step, batch_size=1)
    print('-- Loading Images')
    for _, (image, _) in enumerate(loader):
        pool.add(image)
        
    print('-- Start Generating Images')
    idx = 0
    count = 0
    while True:
        images = pool()
        generated = net(variable(images))

        start = (config.pool_size - config.sample_step)//2
        end = start + config.sample_step
        for i in range(start, end):
            image_a = images[0, :, i, :, :]
            image_b = generated.data[0, :, i, :, :].cpu()
            sample = torch.cat([image_a, image_b], dim=2)
            sample = utils.denorm(sample)

            if not os.path.isdir(config.result_dir):
                os.mkdir(config.result_dir)
            
            name = os.path.join(config.result_dir, 'Epoch-%d-%05d.jpg' % (epoch, count))
            torchvision.utils.save_image(sample, name)
            count += 1
        
        if not pool.check():
            break

        if idx % config.log_frequency == 0:
            speed = (time.time() - step_time) / config.log_frequency
            step_time = time.time()
            format_str ='Generating Images: Step %d Speed %.2f sec/frame'
            print(format_str % (idx, speed))

        idx += 1
