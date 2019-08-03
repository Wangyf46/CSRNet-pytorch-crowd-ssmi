import os
import json
import time
import argparse
import shutil
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision import transforms
from src.model import CSRNet
from src.utils import *
from src.dataset import *

parser = argparse.ArgumentParser(description = "PyTorch CSRNet")
parser.add_argument("--train_json", default = "JSON_320_80/part_B_train.json", 
                    help = "path to train json")
parser.add_argument("--val_json", default = "JSON_320_80/part_B_val.json", 
                    help = "path to val json")
parser.add_argument("--gpu", default = "5,6", type = str, 
                    help = "GPU id to use.")
parser.add_argument("--task", default ="multi_gpu", type = str, 
                    help = "task id to use.")
parser.add_argument("--pre", '-p', default = None, type = str, 
                    help = "path to pretrained model")              ## TODO
parser.add_argument("--lr", default = 1e-6, type = float, 
                    help = "original learning rate")                 ## TODO
parser.add_argument("--epochs_drop", default = 30, type = int, 
                    help = "epochs_drop")                       
parser.add_argument("--start_epoch", default = 0, type = int, 
                    help = "start epoch")
parser.add_argument("--epochs", default = 400, type = int, 
                    help = "epoch")                                  ## TODO
parser.add_argument("--batch_size", default = 2, type = int, 
                    help = "batch size")                          ## TODO
parser.add_argument("--momentum", default = 0.95, type = float) 
parser.add_argument("--decay", default = 5 * 1e-4, type = float)                                           
parser.add_argument("--workers", default = 4,type = int)                                
parser.add_argument("--print_freq", default = 20, type = int, 
                    help = "show train log information")
args = parser.parse_args()

try:
    from termcolor import cprint
except ImportError:
    cprint = None

## create log for ssh check
localtime = time.strftime("%Y-%m-%d", time.localtime())
log_file = open("./logs/record" + localtime + ".txt", 'w')

## create tensorboard dir
logDir = "./tblogs/0319"
if os.path.exists(logDir):
    shutil.rmtree(logDir)  # remove recursive dir
writer = SummaryWriter(logDir)        

def log_print(text, color = None, on_color = None, 
              attrs = None, log_file = log_file):
    print(text, file = log_file)
    if cprint is not None:
        cprint(text, color = color, on_color = on_color, attrs = attrs)
    else:
        print(text)


def main():
    global best_mae, args
    best_mae = 1e6
    args = parser.parse_args()

    ## .json(str) convert to list, list is the image path
    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)  # 320
    with open(args.val_json, 'r') as outfile:
        val_list = json.load(outfile)  # 80

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = time.time()
    torch.cuda.manual_seed(seed)
    model = CSRNet()                     # [weight, bias] Initialize
   
    model = nn.DataParallel(model)
    model = model.cuda()
    #criterion = nn.MSELoss(reduction='sum').cuda()  # TODO
    criterion = SANetLoss(1).cuda()   
    optimizer = torch.optim.SGD(model.parameters(), args.lr, 
                                momentum = args.momentum, 
                                weight_decay = args.decay)
    ##  pre-train model
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_mae = checkpoint['best_mae']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                 .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
 
    ## train
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train_loss = train(train_list, model, criterion, optimizer, epoch)
        val_loss, val_mae = validate(val_list, model, epoch, criterion)
        writer.add_scalar("/train_loss", train_loss, epoch)
        writer.add_scalar("/val_loss", val_loss, epoch)
        is_best = val_mae < best_mae
        best_mae = min(val_mae, best_mae)
        print(epoch, val_mae, best_mae)
        ## save model 1-400
        save_checkpoint({"epoch": epoch + 1,             
                         "arch": args.pre,
                         "state_dict": model.state_dict(),
                         "best_mae": best_mae,
                         "optimizer": optimizer.state_dict(), },
                          is_best, 
                          args.task)

def adjust_learning_rate(optimizer, epoch):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''

    factor = 0.1 ** (epoch // args.epochs_drop)
    args.lr = args.lr * factor
    # print(len(optimizer.param_groups))  1
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


def train(train_list, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform1 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])
    DatasetLoader_train = listDataset(train_list, 
                                     shuffle = True, 
                                     transform = transform1, 
                                     train = True,
                                     batch_size = args.batch_size, 
                                     num_workers = args.workers)
    train_loader = torch.utils.data.DataLoader(DatasetLoader_train, 
                                               batch_size = args.batch_size)
    log_text = "epoch %d, processed %d samples, lr % .10f "\
               %(epoch, epoch * len(train_loader.dataset), args.lr)
    log_print(log_text, color = "green", attrs = ["bold"])
 
    model.train()
    end = time.time()
    for i, (img, gt_density_map) in enumerate(train_loader):
        ##  measure batch_size data loading time
        data_time.update(time.time() - end)
        ## img, gt_density_map from CPU RAM to cuda(GPU)
        img = img.cuda()     # torch.Size([batch_size,3,H,W] 
        img = Variable(img)  # torch.FloatTensor to Variable
        gt_density_map = gt_density_map.type(torch.FloatTensor).unsqueeze(1)
        gt_density_map = gt_density_map.cuda() #torch.size([batch_size,1,H/8,W/8])
        gt_density_map = Variable(gt_density_map) 
       # print(img.shape)
       # print(gt_density_map.shape)

        ## compute output, forward
        et_density_map = model(img)
       # print(et_density_map.shape)

        ## compute loss
        loss = criterion(et_density_map, gt_density_map)
        losses.update(loss.item(), img.size(0))  # batch size == img.size[0]
        
        ## compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## measure batch_size data compute time
        batch_time.update(time.time() - end)
        end = time.time()
        
        ## show train log information
        if i % args.print_freq == 0:
            print_str = "Epoch: [{0}][{1}/{2}]\t"\
                        .format(epoch, i, len(train_loader))
            print_str += "Data time {data_time.cur:.3f}({data_time.avg:.3f})\t"\
                         .format(data_time = data_time)
            print_str += "Batch time {batch_time.cur:.3f}({batch_time.avg:.3f})\t"\
                         .format(batch_time = batch_time)
            print_str += "Loss {loss.cur:.4f}({loss.avg:.4f})\t"\
                         .format(loss = losses)
            #print(print_str)
            log_print(print_str, color = "green", attrs = ["bold"])
    return losses.avg


def validate(val_list, model, epoch, criterion):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform1 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean, std)])
    DatasetLoader_val = listDataset(val_list, 
                                    shuffle = False, 
                                    transform = transform1, 
                                    train = False)
    val_loader = torch.utils.data.DataLoader(DatasetLoader_val, 
                                             batch_size = args.batch_size)
    log_text = "epoch %d, processed %d samples, lr % .10f "\
               %(epoch, epoch * len(val_loader.dataset), args.lr)
    log_print(log_text, color = "red", attrs = ["bold"])
    
    model.eval()
    end = time.time()
    mae = 0
    for i, (img, gt_density_map) in enumerate(val_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        img = Variable(img)                             
        gt_density_map = gt_density_map.type(torch.FloatTensor).unsqueeze(1)
        gt_density_map = gt_density_map.cuda()       
        gt_density_map = Variable(gt_density_map) 
        et_density_map = model(img)
        loss = criterion(gt_density_map, et_density_map)        
        losses.update(loss.item(), img.size(0))
        batch_time.update(time.time() - end)
        mae += abs(et_density_map.data.sum() - gt_density_map.sum()) #todo
        end = time.time()
        if i % args.print_freq == 0:
            print_str = "Epoch: [{0}][{1}/{2}]\t"\
                        .format(epoch, i, len(val_loader))
            print_str += "Data time {data_time.cur:.3f}({data_time.avg:.3f})\t"\
                         .format(data_time = data_time)
            print_str += "Batch time {batch_time.cur:.3f}({batch_time.avg:.3f})\t"\
                         .format(batch_time = batch_time)
            print_str += "Loss {loss.cur:.4f}({loss.avg:.4f})\t"\
                        .format(loss = losses)
            #print(print_str)
            log_print(print_str, color = "red", attrs = ["bold"])
    mae = mae / len(val_loader)
    return losses.avg, mae


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur, n = 1):
        self.cur = cur
        self.sum += cur * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()

      
