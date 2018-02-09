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
def NLL(y,p):
    return -np.sum(y*np.log(softmax(p)))

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
    print("W0 Shape:")
    print(W0.shape)
    b0 = snapshot["b0"].reshape((300,1))
    print("b0 Shape:")
    print(b0.shape)
    W1 = snapshot["W1"]
    print("W1 Shape:")
    print(W1.shape)
    b1 = snapshot["b1"].reshape((10,1))
    print("b1 Shape:")
    print(b1.shape)

    #Load one example from the training set, and run it through the
    #neural network
    x = (M["train5"][148:149].T)/255.0
    print("x Shape:")
    print(x.shape)
    L0, L1, output = forward(x, W0, b0, W1, b1)
    #get the index at which the output is the largest
    print("output shape:")
    print(output.shape)
    y = argmax(output)
    print("prediction:")
    print(y)

    ################################################################################
    #Code for displaying a feature from the weight matrix mW
    #fig = figure(1)
    #ax = fig.gca()    
    #heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
    #fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    #show()
    ################################################################################

#-------------------------------Part 2 Function Implementation------------------------------------------------------#
def compute(X,W,b):
    hypothesis = np.matmul(W.T,X)
    hypothesis = hypothesis+b
    hypothesis = softmax(hypothesis)
    return hypothesis

#-------------------------------Part 2 Test------------------------------------------------------#

def testPart2():
    #Load Data
    M = loadmat("mnist_all.mat")
    #Testing computation function
    #Initialize weights and bias to random normal variables
    W = np.random.normal(0,0.2,(784,10))
    b = np.random.normal(0,0.1,(10,1))
    test =(M["train9"][200]/255.0).flatten().T
    print(test.shape)
    print(compute(test,W,b))

#-------------------------------Part 3 Function Implementation------------------------------------------------------#

def grad_NLL(y,p):

    grad = softmax(p)-y
    return grad


#-------------------------------Part 3 Test------------------------------------------------------#

def testPart3():
    #Load some test data from MNIST
    np.random.seed(5)
    M = loadData()
    W = np.random.normal(0,0.2,(784,10))
    b = np.random.normal(0,0.1,(10,1))
    #Finite difference
    h = 0.00001
    #Take 0 for example for testing
    x = ((M["train0"][150].T)/255.0).reshape((784,1))
    #Create true label array
    y = np.array([1,0,0,0,0,0,0,0,0,0])
    y = np.reshape(y,((10,1)))

    #Create empty arrays to include finite differences
    finite_W = np.zeros((10,784))
    finite_W[4][5] = h
    finite_b = np.zeros((10,1))
    finite_b[3] = h
    
    #Compute cost of finite difference:
    finite = ((NLL(y,np.matmul(W.T+finite_W,x)-(b+finite_b)))-(NLL(y,np.matmul(W.T-finite_W,x)-(b-finite_b))))/(2*h)
    print(finite)
    gradient = grad_NLL(y,np.matmul(W.T,x)-b)
    print(gradient)
 
#-------------------------------Part 4 Implementation------------------------------------------------------#

def grad_descent(NLL, grad_NLL, x, y, init_t, b, alpha, iterations):
    EPS = 1e-5
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = iterations
    iter = 0
    while np.linalg.norm(t-prev_t) > EPS and iter < max_iter:
        prev_t=t.copy()
        t-=alpha*grad_NLL(y, compute(x,t,b))
        if iter % 500 == 0:
            print("Iteration: " + str(iter))
            print("Cost: "+str(NLL(y,compute(x,t,b))))
            print("Gradient: " + str(grad_NLL(y,compute(x,t,b))))
        iter += 1
    return t

#-------------------------------Part 4 Test------------------------------------------------------#

def testPart4():
    #Load data
    M = loadData()
    #Initialize training data
    data = M["train0"]
    #Initialize labels
    labels = np.zeros((10,data.shape[0]))
    labels[0,:]=1
    #Get training data and labels
    i = 1
    while i<10:
        next = M["train"+str(i)]
        new_labels = np.zeros((10,next.shape[0]))
        new_labels[i,:]=1
        data = np.vstack((data,next))
        labels = np.hstack((labels,new_labels))
        i+=1
    data = np.transpose(data)
    
    
    W = np.random.normal(0,0.2,(784,60000))
    b = np.random.normal((60000,60000))
    w_ = grad_descent(NLL,grad_NLL,data,labels,W,b,0.00001, 30000)

#Main Function

def main():
    #testPart2()
    #testPart3()
    #test()
    testPart4()

    
    




if __name__ == "__main__":
    main()



    

