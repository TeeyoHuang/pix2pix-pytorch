import argparse
import os
import numpy as np
import time
import datetime
import sys

import torch
from torch.autograd import Variable

from models import Create_nets
from datasets import Get_dataloader
from options import TrainOptions
from optimizer import *
from utils import sample_images , LambdaLR




#load the args
args = TrainOptions().parse()
# Calculate output of image discriminator (PatchGAN)
D_out_size = 256//(2**args.n_D_layers) - 2
print(D_out_size)
patch = (1, D_out_size, D_out_size)

# Initialize generator and discriminator
generator, discriminator = Create_nets(args)
# Loss functions
criterion_GAN, criterion_pixelwise = Get_loss_func(args)
# Optimizers
optimizer_G, optimizer_D = Get_optimizers(args, generator, discriminator)

# Configure dataloaders
train_dataloader,test_dataloader,_ = Get_dataloader(args)


# ----------
#  Training
# ----------
prev_time = time.time()
#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
for epoch in range(args.epoch_start, args.epoch_num):
    for i, batch in enumerate(train_dataloader):

        # Model inputs
        real_A = Variable(batch['A'].type(torch.FloatTensor).cuda())
        real_B = Variable(batch['B'].type(torch.FloatTensor).cuda())

        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(np.ones((real_A.size(0), *patch))).cuda(), requires_grad=False)
        fake = Variable(torch.FloatTensor(np.zeros((real_A.size(0), *patch))).cuda(), requires_grad=False)

        # Update learning rates
        #lr_scheduler_G.step(epoch)
        #lr_scheduler_D.step(epoch)
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()

        #loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        #print("pred_fake: ",pred_fake.size(),"valid: ", valid.size())
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + args.lambda_pixel * loss_pixel
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = args.epoch_num * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch%d/%d]-[Batch%d/%d]-[Dloss:%f]-[Gloss:%f, loss_pixel:%f, adv:%f] ETA:%s" %
                                                        (epoch+1, args.epoch_num,
                                                        i, len(train_dataloader),
                                                        loss_D.data.cpu(), loss_G.data.cpu(),
                                                        loss_pixel.data.cpu(), loss_GAN.data.cpu(),
                                                        time_left))

        # If at sample interval save image
        if batches_done % args.sample_interval == 0:
            sample_images(generator, test_dataloader, args, epoch, batches_done)


    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), '%s/%s/generator_%d.pth' % (args.model_result_dir,args.dataset_name, epoch))
        torch.save(discriminator.state_dict(), '%s/%s/discriminator_%d.pth' % (args.model_result_dir,args.dataset_name, epoch))
