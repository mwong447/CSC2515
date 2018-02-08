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
import _pickle as cPickle
from scipy.io import loadmat

#Load the data
def loadData():
    M = loadmat("mnist_all.mat")
    return M

#Softmax function
def softmax(y):
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

#Tanh function
def tanh(y,W,b):
    return np.tanh(np.dot(W.T,y)+b)

#Negative log-loss
def NLL(y,y_):
    return np.sum(y_*np.log(y))

#Forward propagation
def forward(x, W0, b0, W1, b1):
    L0 = tanh(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

#Incomplete function for computing the gradient of the cross-entropy cost function w.r.t the parameters of a neural network
def deriv_multilayer(W0, b0, W1, b1, x, L1, y,y_):
    dCdL1 = y-y_
    dCdW1 = np.dot(L0, dCdL1.T)

#-------------------------------Part 2 Implementation------------------------------------------------------#
def compute(X,W,b):
    hypothesis = np.matmul(W.T,X)
    hypothesis = hypothesis+b
    hypothesis = softmax(hypothesis)
    return hypothesis

def testPart2():
    #Load Data
    M = loadmat("mnist_all.mat")
    #Testing computation function
    #Initialize weights and bias to random normal variables
    W = np.random.normal(0,0.2,(784,1))
    b = np.random.normal(0,0.1,(10,1))
    test = M["train9"][200].flatten()/255.0
    print(test.shape)
    print(compute(test,W,b))

#-------------------------------Part 3 Implementation------------------------------------------------------#

def NLL_grad():






#Main Function
def main():
    testPart2()
    
    




if __name__ == "__main__":
    main()



    

