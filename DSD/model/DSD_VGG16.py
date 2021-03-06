# '''
# Define the structure of model VGG16 and VGG19 with batch normalization
# '''
# import torch.nn as nn
# import math
# import torch
# from utils import Sparse_conv

# class DSD_VGG16(nn.Module):

    
#     def __init__(self, features,num_classes=1000, init_weights=True,sparse_ratio = 0.4):
#         super(DSD_VGG16,self).__init__()
#         self.sparse_ratio = sparse_ratio
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             DSD_drop(sparse_ratio = sparse_ratio),
#             nn.Linear(4096,4096),
#             nn.ReLU(True),
#             DSD_drop(sparse_ratio = sparse_ratio),
#             nn.Linear(4096,num_classes)
#         )
#         if init_weights:
#             self._initialize_weights()
    
#     def forward(self,x):
#         x = self.features(x)
#         x = x.view(x.size(0),-1)
#         x = self.classifier(x)
#         return x
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m,nn.Conv2d):
#                 n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))  #intialize weights with gaussian distribution
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m,nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_() 

# def make_layers(cfg, batch_norm=False,sparse_ratio=0.4):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':  #max-pooling 
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:          
#             conv2d = Sparse_conv.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]

#                 # layers += [DSD_drop(sparse_ratio = sparse_ratio)]
#             in_channels = v
#     return nn.Sequential(*layers)    

# #architechture configuration of VGG
# cfg = {
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     # 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }


# def DSD_VGG(**kwargs):
#     """
#     VGG 16-layer model with configuration :'D'
#     """
#     model = DSD_VGG16(make_layers(cfg['D']),**kwargs)
#     return model

# # def vgg19(**kwargs):
# #     """
# #     VGG 19-layer model with configuration :'E'
# #     """
# #     model = VGG(make_layers(cfg['E']),**kwargs)
# #     return model

# class DSD_drop (nn.Module):
#     """Use different phase to control train model densely and sparsely"""
#     (Dense,Sparse) = (0,1)
#     phase = Sparse

#     def __init__(self,sparse_ratio = 0.4):
#         super(DSD_drop,self).__init__()
#         self.sparse_ratio = sparse_ratio

#     def forward(self,x):
#         if self.phase is DSD_drop.Dense:
#             return x
#         else :
#         # get the number of sparse*size :k 
#             one_dim_x = torch.abs(x.reshape(-1,))
#             k  = int(self.sparse_ratio*len(one_dim_x.size()))
#         # get k-th smallest value and use mask to drop connection
#             kth_value = 0.1
#             # threshold = kth_value[k].item()
#             mask = torch.zeros(x.shape).cuda()
#         # drop the connection whose value is bigger than x[k] (if it is negative, then smaller than x[k])
#             out = torch.where(torch.abs(x)>kth_value,x,mask)
#             return out