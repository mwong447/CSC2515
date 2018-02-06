from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import os
import pickle
from scipy.io import loadmat

#Softmax function
def softmax(y):
    return np.exp(y)/np.tile(np.sum(np.exp(y),0),(len(y),1))

#Tanh function
def tanh(y,W,b):
    return np.tanh(np.dot(W.t,y)+b)

#Negative log-loss
def NLL(y,y_):
    return np.sum(y_*np.log(y))

#Part 2 Implementation
def compute(X,W,b):
    hypothesis = np.matmul(W.T,X)
    hypothesis = hypothesis+b
    return softmax(hypothesis)


#Main Function
def main():
    
    #Loading the data from MNIST
    M = loadmat("mnist_all.mat")
    ##Show a random MNIST training data sample
    #imshow(M["train5"][150].reshape((28,28)),cmap=cm.gray)
    #show()


    #Testing computation function
    #Initial weights and bias to random normal variables

    W = np.random.normal(0,0.2,(784,10))
    b = np.random.normal(0,0.1,(10))
    test = M["train9"][200].flatten()/255.0
    print(test.shape)
    print(compute(test,W,b))
    




if __name__ == "__main__":
    main()



    

