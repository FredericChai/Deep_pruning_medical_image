from __future__ import absolute_import
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
class Logger(object):
    """
    write result to logger
    """
    def __init__(self, file_path, title=None): 
        self.file = None
        self.title = '' if title == None else title
        if file_path is not None:
            self.file = open(file_path, 'w')
    '''
    initialize names in the first line of log 
    '''
    def set_names(self, names):
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    '''
    append learning rate, train loss etc. to the tile of log
    '''
    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    #plot the figure
    def plot_metric(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers

        for _, name in enumerate(names):
            if name == "Train Acc" or name == "Valid Acc":
                x = np.arange(len(numbers[name]))
                plt.figure(1)
                plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in ["Train Acc","Valid Acc"]])
        plt.grid(True)

    def plot_train_loss(self,names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            if name == "Trainning Loss":
                plt.figure(2)
                x = np.arange(len(numbers[name]))
                plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(train loss )' ])
        plt.grid(True)

    def plot_valid_loss(self,names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            if name == "Trainning Loss" or name == "Valid Loss":
                plt.figure(3)
                x = np.arange(len(numbers[name]))
                plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '( valid loss)' for name in ["Train Acc","Valid Loss"]])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

def plot_result(logger,result_path):
    logger.plot_metric()
    savefig(result_path+'/log_metric.png')
    logger.plot_train_loss()
    savefig(result_path+'/log_train_loss.png')
    logger.plot_valid_loss()
    savefig(result_path+'/log_valid_loss.png')
    # logger.plot_valid_loss()
    return
                    
#save the plot figure
def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
