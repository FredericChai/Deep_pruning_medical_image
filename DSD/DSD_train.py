from __future__ import print_function
import os
import argparse
import math
import random
import time

import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import classification_report
from tqdm import tqdm

from model import VGG,DSD_VGG16
from utils import *

parser = argparse.ArgumentParser(description = 'Pytorch VGG on PET-CT image')
#path of dataset
parser.add_argument('--dataset',default = '',type = str,help = 'path of dataset')
#configuration 
parser.add_argument('--epochs',default = 100,type=int,help='epochs for each sparse model to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='set epoch number to restart the model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')                 
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, default=200,help='Set epochs to decrease learning rate.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
#Checkpoints
parser.add_argument('-c', '--checkpoint', default='pruned_checkpoint/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the checkpoint to be optimized')
#Architechture for VG G
parser.add_argument('--arch',default='vgg16',type=str,help='VGG Architechture to load the checkpoint')

#Meter for measure()
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate_path',type =str,help = 'path of best model to evaluate the result')
#Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--source_checkpoint', default = '', help = 'path to save the pruned parameters as checkpoint')
parser.add_argument('--sparse_ratio',type = float,help = 'percentage of weights remain')
parser.add_argument('--sparse_epoch',default = 3, type =int, help= 'epochs for sparse trainning the model')
parser.add_argument('--zoom_ratio', type = float,  help = 'ratio for zooming out the weight')
parser.add_argument('--evaluate_mode',default = 0,type =int,help = 'use to evaluate mode')
# parser.add_argument('--norm_activation',dest = 'norm_activation',action = 'store_true')
# parser.add_argument('--filter_prune',dest = 'filter_prune',action = 'store_true')


#load all args
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

#Define the id of device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

#Set Random seed for 
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)



def main():
    # if use sparse iterative train method, then set the sparse ratio
    if not args.sparse_ratio is None:
        if args.evaluate_mode ==1:
            write_result()
            return
        #load the data
        print('loading data:',args.dataset+'Data')
        dataloaders,testDataloaders,dataset_sizes,class_names,numOfClasses = load_data(args.dataset+'Data')
        
        best_acc = 0
        start_epoch = 0

        result_path = args.dataset+'imageclef_pruned_checkpoint/'+str(args.sparse_ratio)+'-'+args.arch+'-epoch'+str(args.epochs)

        make_path(result_path) 
        sparse_checkpoint_path =  os.path.join(result_path,'sparse_checkpoint.pth.tar')
        NN = load_model(args.arch,numOfClasses)
        NN = torch.nn.DataParallel(NN).cuda() 
        cudnn.benchmark = True
        # print('  Total parameters: %.2f' % (sum(p.numel() for p in NN.parameters())))

        #loss funcion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(NN.parameters(), lr=args.lr, momentum=args.momentum)

        title = 'PET-CT-'+args.arch
        #load checkpoint
        checkpoint = torch.load(args.source_checkpoint)
        print('loading checkpoint :'+args.source_checkpoint)
        NN.load_state_dict(checkpoint['state_dict'],strict = True)
        threshold_list = get_threshold(NN.state_dict(),args.sparse_ratio,args.arch)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(result_path, 'log.txt'),title = title)
        logger.set_names(['Learning Rate','Trainning Loss','Valid Loss','Train Acc','Valid Acc'])

        # Train and validate model
        for epoch in range(start_epoch,args.epochs): 
            # adjust_learning_rate(optimizer,epoch)
            adjust_learning_rate(optimizer,epoch,args.lr,args.schedule,args.gamma)
            for parameter_group in optimizer.param_groups:
                current = parameter_group['lr']
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, current))

            if epoch > 0 and epoch < 100:
            	checkpoint = torch.load(result_path+'/checkpoint.pth.tar')
            	# print('begin to prune:',result_path+'/checkpoint.pth.tar')
            	NN.load_state_dict(checkpoint['state_dict'])
            #load the pruned NN
            val_loss, val_acc = validate(dataloaders['val'], NN, criterion, epoch,use_cuda)
            train_loss, train_acc = train(dataloaders['train'], NN, criterion, optimizer, epoch,use_cuda)
            # append logger file
            logger.append([args.lr, train_loss, val_loss, train_acc, val_acc])

            # save model
            is_best = val_acc > best_acc 
            best_acc = max(val_acc, best_acc)
            state = dsd_prune(NN.state_dict(),args.sparse_ratio,args.arch,threshold_list)
            save_checkpoint({
            	    'epoch':epoch,
                    'state_dict': state,
                    'best_acc': best_acc,
                }, is_best,checkpoint=result_path)
            # print('finish to save:',result_path)
        logger.close()
        logger.plot()
        savefig(os.path.join(result_path, 'log.eps'))
        print('Best acc:',best_acc)
        evaluate_checkpoint = torch.load(os.path.join(result_path,'model_best.pth.tar'))
        NN.load_state_dict(evaluate_checkpoint['state_dict'])
        best_epoch = evaluate_checkpoint['epoch']

        report,predict,target = evaluate(testDataloaders, NN, criterion, class_names, use_cuda)
        #write down classfication report
        writer = open(os.path.join(result_path,'classification_report.txt'),'w')
        writer.write(report+'\n')
        writer.write('best_epoch:'+str(best_epoch)+'\n')
        writer.write('best_acc:'+str(best_acc))
        # writer.write('train_schedule: epoch 0-%d,lr:%f   epoch %d-%d,lr:%f  ' % 
        #             (args.schedule,args.lr,args.schedule,args.epochs,args.lr*args.gamma*args.gamma))
        writer.close()
        # write down predict result
        predict_writer = open(os.path.join(result_path,'best_predict.txt'),'w')
        predict_writer.write(predict)
        predict_writer.close()
        #write down target result
        target_writer = open(os.path.join(result_path,'target.txt'),'w')
        target_writer.write(target)
        target_writer.close()  
    # else use the normal train method
    else: 
        normal_train()

        return

def write_result():
    dataloaders,testDataloaders,dataset_sizes,class_names,numOfClasses = load_data(args.dataset+'Data')
    result_path = args.dataset+args.checkpoint+str(args.sparse_ratio)+'-'+args.arch+'-epoch'+str(args.epochs)
    NN = load_model(args.arch,numOfClasses)
    NN = torch.nn.DataParallel(NN).cuda() 
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(NN.parameters(), lr=args.lr, momentum=args.momentum)
    evaluate_checkpoint = torch.load(os.path.join(args.source_checkpoint,'model_best.pth.tar'))
    NN.load_state_dict(evaluate_checkpoint['state_dict'])
    best_epoch = evaluate_checkpoint['epoch']
    report,predict,target = evaluate(testDataloaders, NN, criterion, class_names, use_cuda)
    #write down classfication report
    writer = open(os.path.join(result_path,'classification_report.txt'),'w')
    writer.write(report+'\n')
    writer.write('best_epoch:'+str(best_epoch)+'\n')
    # writer.write('best_acc:'+str(best_acc))
    # writer.write('train_schedule: epoch 0-%d,lr:%f   epoch %d-%d,lr:%f  ' % 
    #             (args.schedule,args.lr,args.schedule,args.epochs,args.lr*args.gamma*args.gamma))
    writer.close()

def normal_train():
    start_epoch = args.start_epoch 
    best_acc = 0 
    #control the path of result
    result_path = os.path.join(args.dataset,'checkpoint/'+args.arch+'-checkpoint-epoch'+str(args.epochs))     
    make_path(result_path) 
    #load the data
    dataloaders,testDataloaders,dataset_sizes,class_names,numOfClasses = load_data(args.dataset+'/Data')
    #build model with defined architechture
    if args.arch.endswith('16'):
        NN = VGG.vgg16(num_classes = numOfClasses)
    elif args.arch.endswith('19'):
        NN = VGG.vgg19(num_classes = numOfClasses)
    elif args.arch.endswith('16_pretrained'):
        NN = models.vgg16(pretrained = True)
        num_features = NN.classifier[6].in_features  
        NN.classifier[6] = nn.Linear(num_features,numOfClasses)  #change last fc layer and keep all other layer if used pretrained model
    elif args.arch.endswith('101_pretrained'):
        NN = models.resnet101(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)
    elif args.arch.endswith('resnet18_pretrained'):
        NN = models.resnet18(pretrained = True)
        num_features = NN.fc.in_features
        NN.fc = nn.Linear(num_features,numOfClasses)

    NN = torch.nn.DataParallel(NN).cuda() 
    cudnn.benchmark = True
    # print('  Total parameters: %.2f' % (sum(p.numel() for p in NN.parameters())))

    #loss funcion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(NN.parameters(), lr=args.lr, momentum=args.momentum)

    title = 'PET-CT-'+args.arch
    #load checkpoint
    logger = Logger(os.path.join(result_path, 'log.txt'),title = title)
    logger.set_names(['Learning Rate','Trainning Loss','Valid Loss','Train Acc','Valid Acc'])

    

    # Train and validate model
    for epoch in range(start_epoch,args.epochs): 
        # adjust the learning rate when the epoch is in the schdule
        adjust_learning_rate(optimizer,epoch,args.lr,args.schedule,args.gamma)
        for parameter_group in optimizer.param_groups:
            current = parameter_group['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, current ))

        train_loss, train_acc = train(dataloaders['train'], NN, criterion, optimizer, epoch,use_cuda)
        val_loss, val_acc = validate(dataloaders['val'], NN, criterion, epoch,use_cuda)

        # append logger file
        logger.append([args.lr, train_loss, val_loss, train_acc, val_acc])

        # save model
        is_best = val_acc > best_acc 
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': NN.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=result_path)

    logger.close()
    logger.plot()
    savefig(os.path.join(result_path, 'log.eps'))
    print('Best acc:',best_acc)
    #evaluate the model based on the best model on val set
    evaluate_checkpoint = torch.load(os.path.join(result_path,'model_best.pth.tar'))
    NN.load_state_dict(evaluate_checkpoint['state_dict'])
    # optimizer.load_state_dict(evaluate_checkpoint['optimizer'])
    best_epoch = evaluate_checkpoint['epoch']

    report,predict,target = evaluate(testDataloaders, NN, criterion, class_names, use_cuda)
    #write down classfication report
    writer = open(os.path.join(result_path,'classification_report.txt'),'w')
    writer.write(report+'\n')
    writer.write('best_epoch:'+str(best_epoch)+'\n')
    writer.write('best_acc:'+str(best_acc))
    writer.close()
    # write down predict result
    predict_writer = open(os.path.join(result_path,'best_predict.txt'),'w')
    predict_writer.write(predict)
    predict_writer.close()
    #write down target result
    target_writer = open(os.path.join(result_path,'target.txt'),'w')
    target_writer.write(target)
    target_writer.close()

    # end the iterative sparse train 
    return

def train(trainloader,model,criterion,optimizer,epoch,use_cuda):
    #train mode
    model.train()
    #metrics of the model
    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    with tqdm(total = len(trainloader)) as pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # measure data loading time
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()   
            pbar.set_description('loss: %.4f top1: %.4f' % (loss.view(-1).data.tolist()[0],top1.avg))
            pbar.update(1)
    return (losses.avg, top1.avg)

def validate(testloader, model, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        torch.no_grad()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)

def evaluate(testloader,model,criterion,class_names,use_cuda):

    #evaluate mode
    model.eval()
    pred = []
    targ = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        torch.no_grad()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # record prediction and target 
        _,output = torch.max(outputs.data,1)
        pred += output.tolist()
        targ += targets.tolist()

    pred_str = ' '.join([str(i) for i in pred])
    targ_str = ' '.join([str(i) for i in targ])
    # sensitivity, F1-score
    report = classification_report(targ,pred,digits = 4,target_names =class_names)

    return report,pred_str,targ_str
 


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def load_data(path):
    #Load data and augment train data
    data_transforms = {
        #Augment the trainning data
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), #crop the given image
            transforms.RandomHorizontalFlip(),  #horizontally flip the image
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #Scale and normalize the validation data
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        #Scale and normalize the validation data
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

    data_dir = path

    image_datasets = {
            x : datasets.ImageFolder(os.path.join(data_dir,x),
                                 data_transforms[x])
            for x in ['train','val','test'] 
        }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],     
                                                        batch_size=32, 
                                                        shuffle=True,
                                                        num_workers=0) 
                                                    for x in ['train','val']}

    testImageLoaders = torch.utils.data.DataLoader(image_datasets['test'],batch_size=16,shuffle=False)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    numOfClasses = len(class_names)
    print('numOfClasses',numOfClasses)

    return dataloaders,testImageLoaders,dataset_sizes,class_names,numOfClasses


def adjust_learning_rate(optimizer,epoch,learning_rate,schedule,gamma):
    if epoch == schedule:
        new_learning_rate = gamma*learning_rate
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] = new_learning_rate


def load_model(model_arch,numOfClasses):
        if args.arch.endswith('16'):
            NN = VGG.vgg16(num_classes = numOfClasses)
        elif args.arch.endswith('19'):
            NN = VGG.vgg19(num_classes = numOfClasses)
        elif args.arch.endswith('vgg16_pretrained'):
            NN = models.vgg16(pretrained = True)
            num_features = NN.classifier[6].in_features  
            NN.classifier[6] = nn.Linear(num_features,numOfClasses)  
#change last fc layer and keep all other layer if used pretrained model
        elif args.arch.endswith('resnet101_DSD'):
            NN = models.resnet101()
            num_features = NN.fc.in_features
            NN.fc = nn.Linear(num_features,numOfClasses)
        elif args.arch.endswith('resnet18'):
            NN = models.resnet18(pretrained = True)
            num_features = NN.fc.in_features
            NN.fc = nn.Linear(num_features,numOfClasses)
        elif args.arch.endswith('vgg16_DSD'):
            NN = VGG.vgg16(num_classes = numOfClasses)
        elif args.arch.endswith('resnet101_DSD'):
            NN = models.resnet101()
            num_features = NN.fc.in_features
            NN.fc = nn.Linear(num_features,numOfClasses)
        elif args.arch.endswith('resnet34_DSD'):
            NN = models.resnet34()
            num_features = NN.fc.in_features
            NN.fc = nn.Linear(num_features,numOfClasses)
        elif args.arch.endswith('resnet18_DSD'):
            NN = models.resnet18()
            num_features = NN.fc.in_features
            NN.fc = nn.Linear(num_features,numOfClasses)
        elif args.arch.endswith('resnet50_DSD'):
            NN = models.resnet50()
            num_features = NN.fc.in_features
            NN.fc = nn.Linear(num_features,numOfClasses)
        elif args.arch.endswith('resnet101_DSD'):
            NN = models.resnet101()
            num_features = NN.fc.in_features
            NN.fc = nn.Linear(num_features,numOfClasses)

        return NN

if __name__ == '__main__':
    main()
