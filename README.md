# Deep_pruning_medical_image
This repository provides the implementation of the network pruning method. The model (e.g. ResNet) will firstly be pretrained on ImageNet and then finetuned on medical datasets(e.g. ImageCLEF2016). Then we will prune CNN neurons and finetune the model agian. It also implement dense-sparse-dense training method from paper [Dsd: Dense-sparse-dense training for deep neural networks] I further impelment a method named iterative sparsity training (IST) method based on DSD.

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Pytorch](https://pytorch.org/) (Recommended version 9.2)
- [Python 3](https://www.python.org/)

## Quick Start
[main.py](main.py) and receipes provides an example of our pruning approach. In this example, we prune a  sophistcated convolutional neural network (ResNet50). 

The method takes two parameters:
1. Number of pruning iterations 
2. Percentage of neurons to be removed in each iteration 

## Results
Tables below show the comparison between network pruning results with based on different pruning methods. 

ResNet50 on ImageCLEF2016

|     Method     | Compression Rate (%) | Accuracy ↓ (%) |
|:--------------:|:-----:|:----------------:|
| Baseline (Finetuning) |  0 |       91.03       |
| Prune (norm of weight)| 40% |  90.4       |
| Prune (norm of weight) |   30% | 90.2 |

ResNet50 on ImageCLEF2016

|     Method     | Accuracy ↓ (%) |
|:--------------:|:----------------:|
| Train from scratch |  77.9    |
| Finetuning)| 85.8  |
| DSD |  86.2 |
| IST  |  86.8 |

Figures below show that weight distribution change in IST method. The first column indicate that the weight distribution from finetuning follows the normal distribution. The distribution in the second column of indicates that unimportant parameters  was removed. The third column indicate taht the removed weights are recovered. 
![image](https://github.com/FredericChai/Deep_pruning_medical_image/blob/main/Result/Parameters%20distribution.png)



