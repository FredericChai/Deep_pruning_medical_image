import shutil
import torch
import sys
import os
import sklearn 
import re
import math
import numpy

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accracy_k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def make_path(path):
    """make dirs for checkpoint"""
    if not os.path.isdir(path):
        os.makedirs(path)


def prune(load_path,sparse_ratio,arch):
    checkpoint = torch.load(load_path)
    state_dict = checkpoint['state_dict']
    
    if arch.startswith('vgg16'):
        for k in state_dict:
            if not k.startswith('module.features.0') and k.endswith('weight'):
                #compute the threshold
                one_dim_tensor = torch.abs(state_dict[k].reshape(-1,)) 
                threshold_index = int(sparse_ratio*len(one_dim_tensor))
                sorted_tensor,_ = one_dim_tensor.sort(0,descending = True)
                threshold =  sorted_tensor[threshold_index].item()
                #apply threshold to layer
                sparse_tensor = state_dict[k]
          
                mask = torch.zeros(sparse_tensor.shape).cuda()
                state_dict[k] = torch.where(torch.abs(sparse_tensor)>threshold,sparse_tensor,mask)
    if arch.startswith('resnet'):
        for k in state_dict:
            if re.match(r'module.layer+\d+.+\d+.conv+\d+.weight',k):  # The first layer is not the form of module.layer
                #compute the threshold
                one_dim_tensor = torch.abs(state_dict[k].reshape(-1,)) 
                threshold_index = int(sparse_ratio*len(one_dim_tensor))
                sorted_tensor,_ = one_dim_tensor.sort(0,descending = True)
                threshold =  sorted_tensor[threshold_index].item()
                #apply threshold to layer
                sparse_tensor = state_dict[k]
                mask = torch.zeros(sparse_tensor.shape).cuda()
                state_dict[k] = torch.where(torch.abs(sparse_tensor)>threshold,sparse_tensor,mask)
    return state_dict

def get_threshold(current_checkpoint,sparse_ratio,arch):
    threshold_list = {}
    if arch.startswith('vgg16'):    
        for layer in current_checkpoint:
            if layer.endswith('weight'):
                #compute the threshold
                one_dim_tensor = torch.abs(current_checkpoint[layer].reshape(-1,)) 
                threshold_index = int(sparse_ratio*len(one_dim_tensor))
                sorted_tensor,_ = one_dim_tensor.sort(0,descending = True)
                threshold =  sorted_tensor[threshold_index].item()
                #use dictionary to store threshold
                threshold_list[layer] = threshold
    return threshold_list

def dsd_prune(current_checkpoint,sparse_ratio,arch,threshold_list):
    state_dict = current_checkpoint
    
    if arch.startswith('vgg16'):
        for k in state_dict:
            for j in threshold_list:
                if k==j:
                    threshold = threshold_list[k]
                #compute the threshold
                # one_dim_tensor = torch.abs(state_dict[k].reshape(-1,)) 
                # threshold_index = int(sparse_ratio*len(one_dim_tensor))
                # sorted_tensor,_ = one_dim_tensor.sort(0,descending = True)
                # threshold =  sorted_tensor[threshold_index].item()
                #apply threshold to layer
                    sparse_tensor = state_dict[k]                         
                    mask = torch.zeros(sparse_tensor.shape).cuda()
                    state_dict[k] = torch.where(torch.abs(sparse_tensor)>threshold,sparse_tensor,mask)
                    # print('find the pruned key',k)
    if arch.startswith('resnet101'):
        for k in state_dict:
            if re.match(r'module.layer+\d+.+\d+.conv+\d+.weight',k):  # The first layer is not the form of module.layer
                #compute the threshold
                one_dim_tensor = torch.abs(state_dict[k].reshape(-1,)) 
                threshold_index = int(sparse_ratio*len(one_dim_tensor))
                sorted_tensor,_ = one_dim_tensor.sort(0,descending = True)
                threshold =  sorted_tensor[threshold_index].item()
                #apply threshold to layer
                sparse_tensor = state_dict[k]
                mask = torch.zeros(sparse_tensor.shape).cuda()
                state_dict[k] = torch.where(torch.abs(sparse_tensor)>threshold,sparse_tensor,mask)
    return state_dict

def filter_prune(load_path,sparse_ratio,arch):
    """prune and sparse train the checkpoint according to the average value to filter
    basic idea: if the average value is not big, the filter may not be very big
    """
    checkpoint = torch.load(load_path)
    state_dict = checkpoint['state_dict']
    optimizer = checkpoint['optimizer']
    if arch.startswith('vgg16'):
        for  layer in state_dict:
            if not layer.startswith('module.features.0') and layer.endswith('weight'):
                #if dimension of conv is less than 4, it must be full connect layer
                if len(list(state_dict[layer].shape))<4:
                    one_dim_tensor = torch.abs(state_dict[layer].reshape(-1,)) 
                    threshold_index = int(sparse_ratio*len(one_dim_tensor))
                    try:
                        sorted_tensor,_ = one_dim_tensor.sort(0,descending = True)
                    except RuntimeError as exception:
                        if "out of memory" in str(exception):
                            print('WARRING: OUT OF MEMORY')
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                            else:
                                raise exception
                    threshold =  sorted_tensor[threshold_index].item()
                    #apply threshold to layer
                    sparse_tensor = state_dict[layer]
                    mask = torch.zeros(sparse_tensor.shape).cuda()
                    #normal activation or zero activation                 
                    state_dict[layer] = torch.where(torch.abs(sparse_tensor)>threshold,sparse_tensor,mask)
                else:
                    c,k,r,s = state_dict[layer].shape[0],state_dict[layer].shape[1],state_dict[layer].shape[2],state_dict[layer].shape[3]
                    #get the absoulte avrage value of each filter
                    conv_mean = torch.mean(torch.abs(state_dict[layer]),3)
                    oneDim_conv = torch.mean(conv_mean,2).reshape(c*k)
                    num,index = torch.sort(oneDim_conv,0)
                    #get the sorted index of the vector
                    threshold_index = int(sparse_ratio*c*k)
                    #set filter to zero or random number
                    for i in range(threshold_index):
                        i_th = math.floor(index[i]/k)
                        j_th = (index[i]%k).item()
                        state_dict[layer][i_th][j_th] = torch.zeros(r,s)
    return state_dict,optimizer

def Reduce_prune(load_path,sparse_ratio,arch):
    """prune the model by zooming out the weight
    """
    checkpoint = torch.load(load_path)
    state_dict = checkpoint['state_dict']
    # optimizer  =checkpoint['optimizer']
    if arch.startswith('vgg16'):
        for k in state_dict:
            if not k.startswith('module.features.0') and k.endswith('weight'):
                #compute the threshold
                one_dim_tensor = torch.abs(state_dict[k].reshape(-1,)) 
                threshold_index = int(sparse_ratio*len(one_dim_tensor))
                sorted_tensor,_ = one_dim_tensor.sort(0,descending = True)
                threshold =  sorted_tensor[threshold_index].item()
                #apply threshold to layer
                sparse_tensor = state_dict[k]         
                mask = c*sparse_tensor
                state_dict[k] = torch.where(torch.abs(sparse_tensor)>threshold,sparse_tensor,mask)
    if arch.startswith('resnet101'):
        for k in state_dict:
            if re.match(r'module.layer+\d+.+\d+.conv+\d+.weight',k):  # The first layer is not the form of module.layer
                #compute the threshold
                one_dim_tensor = torch.abs(state_dict[k].reshape(-1,)) 
                threshold_index = int(sparse_ratio*len(one_dim_tensor))
                sorted_tensor,_ = one_dim_tensor.sort(0,descending = True)
                threshold =  sorted_tensor[threshold_index].item()
                #apply threshold to layer
                sparse_tensor = state_dict[k]
                mask = c*sparse_tensor
                state_dict[k] = torch.where(torch.abs(sparse_tensor)>threshold,sparse_tensor,mask)
    
    return state_dict

