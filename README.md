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

|     Method     | Compression Rate (%) | Accuracy ↓ (percentage points) |
|:--------------:|:-----:|:----------------:|
| Baseline (Finetuning) |  0 |       91.03       |
| Prune (norm of weight)| 40% |  90.4       |
| Prune (norm of weight) |   30% | 90.2 |

ResNet34 on ImageCLEF2016

|     Method     | Compression Rate (%) | Accuracy ↓ (percentage points) |
|:--------------:|:-----:|:----------------:|
| Baseline (Finetuning) |  0 |       90.3       |
| Prune (norm of weight)| 40% |  89.8       |
| Prune (norm of weight) |   30% | 89.2 |

ResNet50 on ImageCLEF2016

|     Method     |  | Accuracy ↓ (percentage points) |
|:--------------:|:----------------:|
| Train from scratch |  77.9       |
| Finetuning)| 40% |  85.8       |
| DSD |  86.2 |
| IST  |  86.8 |

