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

#Helper function to load the training data from matrix
def loadTrain(M):
    #Initialize training data
    training = M["train0"]
    #Initialize labels
    labels = np.zeros((10,training.shape[0]))
    labels[0,:]=1
    #Get training data and labels
    i = 1
    while i<10:
        next = M["train"+str(i)]
        new_labels = np.zeros((10,next.shape[0]))
        new_labels[i,:]=1
        training = np.vstack((training,next))
        labels = np.hstack((labels,new_labels))
        i+=1
    training = np.transpose(training)
    training = training/255.0
    return training, labels

#Helper function to display an MNIST data point.
def display(x):
    imshow(x.reshape((28,28)),cmap = cm.gray)
    show()

#Softmax function
def softmax(y):
    return np.exp(y)/np.tile(np.sum(np.exp(y),0), (len(y),1))

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


#-------------------------------Part 2 Function Implementation------------------------------------------------------#
def compute(X,W,b):
    hypothesis = np.matmul(W.T,X)
    hypothesis = hypothesis+b
    return hypothesis

#-------------------------------Part 2 Test------------------------------------------------------#

def testPart2():
    #Load Data
    M = loadData()
    train = loadTrain(M)
    #Testing computation function
    #Initialize weights and bias to random normal variables
    W = np.random.normal(0.0,0.2,(784,10))
    b = np.random.normal(0,0.1,(10,60000))
    hypothesis = compute(train,W,b)
    print(softmax(hypothesis))

#-------------------------------Part 3 Function Implementation------------------------------------------------------#
#Computes the gradient of negative log-loss function
def grad_NLL(y,o):

    p = softmax(o)
    grad = p - y
    return grad

#-------------------------------Part 3 Test------------------------------------------------------#

def testPart3():
    #Load some test data from MNIST
    np.random.seed(5)
    M = loadData()
    W = np.random.normal(0,0.2,(784,10))
    b = np.random.normal(0,0.1,(10,1))
    #Finite difference
    h = 0.000001
    #Take 0 for example for testing
    x = ((M["train0"][150].T)/255.0).reshape((784,1))
    #display(x)
    #Create true label array
    y = np.array([1,0,0,0,0,0,0,0,0,0])
    y = np.reshape(y,((10,1)))

    #Create empty arrays to include finite differences
    finite_W = np.zeros((784,10))
    finite_b = np.zeros((10,1))

    
    #test with only biases
    finite_b[2] = h
    finite = (NLL(y,compute(x, W,b+finite_b))-NLL(y,compute(x,W,b)))/(h)
    print("cost: " + str(finite))
    gradient = grad_NLL(y,compute(x,W,b))
    print("Gradient Vector: ")
    print(gradient)
    print(np.sum(gradient))
    
    finite_W[:,2]=h
    ##Test with only weights
    finite = (NLL(y,compute(x, W+finite_W,b))-NLL(y,compute(x,W,b)))/(h)
    print("Cost: " + str(finite))
    gradient = grad_NLL(y,compute(x,W,b))
    print("Gradient vector:")
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
    data, labels = loadTrain(M)
    print(data.shape)
    print(labels.shape)
    
    
    W = np.random.normal(0,0.2,(784,10))
    b = np.random.normal(0,0.2,(10,60000))
    
    w_ = grad_descent(NLL,grad_NLL,data,labels,W,b,0.00001, 30000)

#Main Function

def main():
    #testPart2()
    #testPart3()
    #test()
    testPart4()

    
    




if __name__ == "__main__":
    main()



    

