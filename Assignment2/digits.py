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
import pickle as pickle
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

#-------------------------------Test Code Implementation------------------------------------------------------#
def test():
    #Load sample weights for the multilayer neural network
    M = loadData()
    file = open(os.path.join(os.getcwd(),'snapshot50.pkl'),'rb')
    snapshot = pickle.load(file,encoding="latin1")
    W0 = snapshot["W0"]
    print(W0.shape)
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))

    #Load one example from the training set, and run it through the
    #neural network
    x = M["train5"][148:149].T    
    L0, L1, output = forward(x, W0, b0, W1, b1)
    #get the index at which the output is the largest
    y = argmax(output)

    ################################################################################
    #Code for displaying a feature from the weight matrix mW
    #fig = figure(1)
    #ax = fig.gca()    
    #heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
    #fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    #show()
    ################################################################################

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





#Main Function

def main():
    #testPart2()

    
    




if __name__ == "__main__":
    main()



    

