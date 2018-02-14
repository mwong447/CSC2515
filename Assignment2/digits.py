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

#Helper function to split data
def split(data,labels, percentage):
    combined = np.vstack((data,labels))
    combined = np.transpose(combined)
    np.random.seed(1)
    np.random.shuffle(combined)
    combined = np.transpose(combined)
    trainData, validData = combined[:-10,:int(percentage*combined.shape[1])], combined[:-10,int(percentage*combined.shape[1]):int(combined.shape[1])]
    trainLabels, validLabels = combined[784:,:int(percentage*combined.shape[1])], combined[784:,int(percentage*combined.shape[1]):int(combined.shape[1])]
        
    return trainData, validData, trainLabels, validLabels


#Helper function to display an MNIST data point.
def display(x):
    imshow(x.reshape((28,28)),cmap = cm.gray)
    show()

#Provided Softmax function
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
    x = (M["train5"][148:200].T)/255.0
    print("x Shape:")
    print(x.shape)
    L0, L1, output = forward(x, W0, b0, W1, b1)
    #get the index at which the output is the largest
    print("Output shape:")
    print(output.shape)
    print("L0 shape:")
    print(L0.shape)
    print("L1 shape:")
    print(L1.shape)
    y = argmax(output,axis=0)
    print("Prediction:")
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
    W = np.random.normal(0.0,0.1,(784,10))
    b = np.random.normal(0,0.1,(10,1))
    hypothesis = compute(train,W,b)
    print(softmax(hypothesis))

#-------------------------------Part 3 Function Implementation------------------------------------------------------#
#Computes the gradient of negative log-loss function for weights only
def grad_NLL_W(y,o,layer):

    p = softmax(o)
    grad = p - y
    grad = np.matmul(grad,np.transpose(layer))
    return np.transpose(grad)


#Computes the gradient of negative log-loss function for biases only
def grad_NLL_b(y,o):
    p = softmax(o)
    grad = p-y
    grad = np.sum(grad,axis=1,keepdims = True)
    return grad

#-------------------------------Part 3 Test------------------------------------------------------#

def testPart3():
    #Test gradient functionality
    y = np.zeros((10,1))
    y[1,:]=1
    
    #Create a test matrix
    M = loadData()
    test0 = ((M["train1"][130].T)/255.0).reshape((784,1))
    W = np.random.normal(0, 0.2, (784,10))
    b = np.random.normal(0,0.2, (10,1))
    #print(np.where(test0 != 0))

    #Create a finite difference
    h = 0.00001

    #Weight testing
    finite_W = np.zeros((784,10))
    finite_W[542,0]=h
    finite_d = (NLL(y,compute(test0,W+finite_W,b))-NLL(y,compute(test0,W,b)))/(h)
    print("Cost for row 542, column 0: " + str(finite_d))
    gradient = grad_NLL_W(y,compute(test0,W,b),test0)
    print("Gradient for row 542, column 0: " + str(gradient[542,0]))

    #Bias testing
    finite_b = np.zeros((10,1))
    finite_b[1,:] = h
    finite_d = (NLL(y,compute(test0,W,b+finite_b))-NLL(y,compute(test0,W,b)))/(h)
    print("Cost for second element in bias: " + str(finite_d))
    gradient = grad_NLL_b(y,compute(test0,W,b))
    print("Gradient matrix: " + str(gradient))

    #Reinitialize test variables for another test
    finite_W = np.zeros((784,10))
    finite_b = np.zeros((10,1))
    y = np.zeros((10,1))
    test1 = ((M["train9"][130].T)/255.0).reshape((784,1))
    y[9,:]=1

    #Weight testing
    finite_W[300,4] = h
    finite_d = (NLL(y,compute(test1,W+finite_W,b))-NLL(y,compute(test1,W,b)))/(h)
    print("Cost for row 300, column 4: " + str(finite_d))
    gradient = grad_NLL_W(y,compute(test1,W,b),test1)
    print("Gradient for row 300, column 4: " + str(gradient[300,4]))

    #Bias testing
    finite_b[4,:] = h
    finite_d = (NLL(y,compute(test1,W,b+finite_b))-NLL(y,compute(test1,W,b)))/(h)
    print("Cost for fifth element in bias: " + str(finite_d))
    gradient = grad_NLL_b(y,compute(test1,W,b))
    print("Gradient matrix: " + str(gradient))

 
#-------------------------------Part 4 Implementation------------------------------------------------------#
def grad_descent(NLL, grad_NLL_W, grad_NLL_b, x, y, init_w, init_b, alpha, iterations):
    iters = list()
    costy = list()
    
    EPS = 1e-5
    prev_w = init_w-10*EPS
    prev_b = init_b-10*EPS
    w = init_w.copy()
    b = init_b.copy()
    max_iter = iterations
    iter = 0
    while np.linalg.norm(w-prev_w) > EPS and iter < max_iter and np.linalg.norm(b-prev_b) > EPS:
        prev_w=w.copy()
        prev_b = b.copy()
        w-=alpha*(grad_NLL_W(y, compute(x,w,b),x))
        b-=alpha*(grad_NLL_b(y,compute(x,w,b)))
        if iter % 10 == 0:
            cost_i = NLL(y,compute(x,w,b))
            iters.append(iter)
            costy.append(cost_i)
            print("Iteration: " + str(iter))
            print("Cost: "+str(cost_i))
        iter += 1

    x_plot = iters
    y1_plot = costy
    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("Iterations vs. Log-loss")
    plt.plot(x_plot, y1_plot, 'r--')
    plt.show()

    return w, b

#-------------------------------Part 4 Test------------------------------------------------------#

def testPart4():
    #Load data
    M = loadData()
    #Initialize training data
    data, labels = loadTrain(M)
    trainData, validData, trainLabels, validLabels = split(data,labels, 0.8)
    print(trainData.shape)
    print(trainLabels.shape)
    
    W = np.random.normal(0.0,0.1,(784,10))
    b = np.random.normal(0, 0.1,(10,1))
        
    w_train, b_train = grad_descent(NLL, grad_NLL_W, grad_NLL_b, trainData, trainLabels, W, b, 0.00001, 400)

    trainPred = compute(trainData, w_train,b_train)
    trainPred = np.argmax(trainPred,axis = 0)
    trainLabels = np.argmax(trainLabels,axis = 0)
    print("Training accuracy: " + str(np.sum(np.equal(trainPred,trainLabels))/(float(trainPred.shape[1]))))

    validPred = compute(validData,w_train,b_train)
    validPred = np.argmax(validPred,axis = 0)
    validLabels = np.argmax(validLabels,axis = 0)

    print("Validation accuracy: " + str(np.sum(np.equal(validPred,validLabels))/(float(validPred.shape[1]))))

#-------------------------------Part 5 Implementation------------------------------------------------------#
def grad_descent_m(NLL, grad_NLL_W, grad_NLL_b, x, y, init_w, init_b, alpha, iterations):
    iters = list()
    costy = list()
    
    EPS = 1e-5
    prev_w = init_w-10*EPS
    prev_b = init_b-10*EPS
    w = init_w.copy()
    b = init_b.copy()
    max_iter = iterations
    iter = 0

    v = np.zeros((784,10))
    momentum_rate = 0.5

    while np.linalg.norm(w-prev_w) > EPS and iter < max_iter and np.linalg.norm(b-prev_b) > EPS:
        prev_w=w.copy()
        prev_b = b.copy()
        grad_desc = alpha*(grad_NLL_W(y, compute(x,w,b),x))
        mv = np.multiply(momentum_rate,v)
        v = np.add(mv, grad_desc)
        w-= v
        b-=alpha*(grad_NLL_b(y,compute(x,w,b)))
        if iter % 100 == 0:
            print("Iteration: " + str(iter))
            print("Cost: "+str(NLL(y,compute(x,w,b))))
            cost_i = NLL(y,compute(x,w,b))
            iters.append(iter)
            costy.append(cost_i)
        iter += 1

    x_plot = iters
    y1_plot = costy
    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("Iterations vs. Log-loss")
    plt.plot(x_plot, y1_plot, 'r--')
    plt.show()
    return w, b


#Main Function

def main():
    np.random.seed(1)
    #testPart2()
    #testPart3()
    #test()
    testPart4()

    
    




if __name__ == "__main__":
    main()



    

