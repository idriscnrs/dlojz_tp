#######################################################################
### LR Finder algorithm based on Grugger's script using smoothed loss: 
### https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
### Method based on Smith's paper (2017): 
### https://arxiv.org/pdf/1506.01186
###
### This script is part of the IDRIS 'DLO-JZ' course
### Author MP
### Date October 2026
#######################################################################
import argparse
import os
import sys
import random
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import math
import csv

# Fix random seeds
random.seed(123)                           
np.random.seed(123)
torch.manual_seed(123) 

def main():
    parser = argparse.ArgumentParser()                                 
    parser.add_argument('-e', '--nepochs', default=20, type=int,  
                        help='The number of epochs (control the discretization of the LR finder)')                     
    parser.add_argument('-b', '--batch-size', default=256, type=int,  
                        help='Batch size per GPU')                     
    parser.add_argument('-o', '--optim', default='SGD', type=str,  
                        help='Optimizer: SGD or AdamW')                     
    parser.add_argument('--min-lr', default=1e-5, type=float,             
                        help='Mininimum learning rate to test')                          
    parser.add_argument('--max-lr', default=1e0, type=float,             
                        help='Maximum learning rate to test')                          
    parser.add_argument('--mom', default=0.9, type=float,              
                        help='Momentum')                               
    parser.add_argument('--wd', default=0., type=float,                
                        help='Weight decay')                           
    args = parser.parse_args()
    train(args)
    

def train(args):
    
    # Define model
    model = models.resnet18(num_classes=10)
    model.name = 'Resnet-18'
    print(f'Model: {model.name}')
    
    # Send to GPU
    gpu = torch.device("cuda")
    model = model.to(gpu, memory_format=torch.channels_last)

    # Define loss function (criterion) 
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)     
    
    # Define optimizer
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), args.max_lr, momentum=args.mom, weight_decay=args.wd)
    elif args.optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), args.max_lr, betas=(args.mom, 0.999), weight_decay=args.wd)
    else:
        sys.exit("Optimizer {args.optim} not recognized, should be SGD or AdamW")
    print(f'Optimizer: {optimizer}')    
        
    # Transformations
    transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),   # Data Augmentation - Horizontal Flip
        transforms.RandomCrop(32, padding=4),# Data Augmentation - Random Crop
        transforms.ToTensor()                # Convert PIL Image to a Tensor
        ])

    # Dataset
    train_dataset = torchvision.datasets.CIFAR10(root=os.environ['DSDIR'], 
                                                 train=True, 
                                                 download=False, 
                                                 transform=transform)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               persistent_workers=True,
                                               pin_memory=True,
                                               prefetch_factor=2)
    
    # Loop initialization (from Gugger's script)
    max_niter = len(train_loader) * args.nepochs
    mult = (args.max_lr / args.min_lr) ** (1/max_niter)
    lr = args.min_lr
    optimizer.param_groups[0]['lr'] = lr
    beta = 0.98 # to compute smoothed loss 
    niter = 0
    avg_loss = 0.
    best_loss = 0.
    losses = []
    log_lrs = []
    loss_exploded = False

    print(f'Batch size: {args.batch_size}')
    print(f'>>> Scanning for min_lr={args.min_lr} to max_lr={args.max_lr}')
    print(f'    Maximum number of iterations (nepochs x nsteps): {max_niter}')
    
    ######## TRAINING #######
    for epoch in range(args.nepochs):
        for i, (images, labels) in enumerate(train_loader):
            
            niter += 1

            # switch to training mode
            model.train()                                                

            # distribution of images and labels to all GPUs
            images = images.to(gpu, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(gpu, non_blocking=True)

            # Reset gradients to None
            optimizer.zero_grad()

            # Runs the forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Compute the smoothed loss (from Gugger's script)
            avg_loss = beta * avg_loss + (1-beta) *loss.item()
            smoothed_loss = avg_loss / (1 - beta**(niter))

            # Stop if the loss is exploding (from Gugger's script)
            if niter > 1 and smoothed_loss > 4 * best_loss: 
                print('Loss is exploding at lr={lr} -> break the loop')
                loss_exploded = True
                break

            # Record the best loss (from Gugger's script)
            if smoothed_loss < best_loss or niter==1:
                best_loss = smoothed_loss

            # Store the values (from Gugger's script)
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Do the optimizer step (from Gugger's script)
            loss.backward()
            optimizer.step()

            # Update the lr for the next step (from Gugger's script)
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr
        
        if loss_exploded: break

        print(f'Epoch {epoch}/{args.nepochs}: lr = {lr}, last loss = {losses[-1]}, best loss = {best_loss}')


    #### WRITE CSV #################
    filename = f"lrfinder_optim{args.optim}_bs{args.batch_size}_minlr{args.min_lr}_maxlr{args.max_lr}_mom{args.mom}_wd{args.wd}_nepochs{args.nepochs}_{os.environ['SLURM_JOBID']}.csv"
    with open(filename , 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(log_lrs)
        wr.writerow(losses)
    print(f'>>> CSV file written: {filename}')

if __name__ == '__main__':
    main()

