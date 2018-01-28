from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imshow
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import math

def getActors(X,Y=None):
    CurrentDirectory = os.getcwd()
    FileName = str(X)
    FileName = os.path.join(CurrentDirectory,X)
    if Y is None:
        act = list(set([a.split("\t")[0] for a in open(FileName).readlines()]))
        print(act)
        return act
    else:
        return FileName

def rgb2gray(rgb):
    try:
        r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
        gray = 0.2989*r+0.5870*g+0.1140*b
        return gray
    except IndexError:
        print("Image is already grayscale")

def getRawData():
    actors = "facescrub_actors.txt"
    list = getActors(actors)
    if not os.path.exists(os.path.join(os.getcwd(),"uncropped/")):
        os.makedirs(os.path.join(os.getcwd(),"uncropped/"))
    for a in list:
        name = a.split()[1].lower()
        i = 0
        for line in open(getActors(actors,1)):
            if a in line:
                filename = name + str(i)+'.'+line.split()[4].split('.')[-1]
                if filename.endswith(".jpg"):
                    try:
                       print("Attempting to look at up URL: " + line.split()[4])
                       if os.path.isfile(os.path.join(os.path.join(os.getcwd(),"uncropped"),filename)):
                           continue
                       page = urllib.request.urlretrieve(line.split()[4],os.path.join(os.path.join(os.getcwd(),"uncropped"),filename),timeout)
                       print("Attempting to write to disk: " + filename)
                       i+=1
                    except urllib.error.HTTPError:
                        print("Unable to get content")
                    except urllib.error.URLError:
                        print("Unable to reach URL")
                    except urllib.error.ContentTooShortError:
                        print("Content too short")

def getCroppedData(list,file, partNumber):
    partString = str("cropped"+str(partNumber)+"/")
    if not os.path.exists(os.path.join(os.getcwd(),partString)):
        os.makedirs(os.path.join(os.getcwd(),partString))
    for a in list:
        name = a.split()[1].lower()
        i=0
        for line in open(getActors(file,1)):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                if filename.endswith(".jpg") and i != 130:
                    try:
                        if os.path.isfile(os.path.join(os.path.join(os.getcwd(),partString),filename)):
                            continue
                        url = line.split()[4]
                        print("Attempting to look at URL:" + url)
                        print("Filename is: " + filename)
                        print("Cropping image...")
                        x1 = int(line.split()[5].split(",")[0])
                        y1 = int(line.split()[5].split(",")[1])
                        x2 = int(line.split()[5].split(",")[2])
                        y2 = int(line.split()[5].split(",")[3])
                        try:
                            img = imread(urllib.request.urlopen(url))
                            img = img[y1:y2,x1:x2]
                            img = img/255.0
                            img = rgb2gray(img)

                            img = imresize(img,[32,32],'nearest')
                            print("Attempting to save file to disk")
                            imsave(os.path.join(os.path.join(os.getcwd(),partString),filename),img)
                            i+=1
                        except IOError:
                            print("Unable to read image")
                        except ValueError:
                            print("Array corrupted")
                    except urllib.error.HTTPError:
                        print("Unable to get content")
                    except urllib.error.URLError:
                        print("Unable to reach URL")
                    except urllib.error.ContentTooShortError:
                        print("Content too short")

def flattenData(x):
    x = np.array(x)
    x = x.flatten()
    return x

def f(x,y,theta):
    ones = np.ones((1,x.shape[1]))
    x = np.vstack((ones,x))
    return np.sum(np.power((y-np.dot(theta.T,x)),2))    

def df(x, y, theta):
    ones = np.ones((1,x.shape[1]))
    x = np.vstack((ones,x))
    return -2*np.sum((y-np.dot(theta.T,x))*x, 1)
    
def grad_descent(f,df,x,y,init_t,alpha, iterations):
    EPS = 1e-5
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = iterations
    iter = 0
    while np.linalg.norm(t-prev_t) > EPS and iter < max_iter:
        prev_t=t.copy()
        t-=alpha*df(x,y,t)
        if iter % 500 == 0:
            print("Iteration: " + str(iter))
            print("Cost: "+str(f(x,y,t)))
            print("Gradient: " + str(df(x,y,t)))
        iter += 1
    return t

def f2(x,y,theta):
    ones = np.ones((1,x.shape[1]))
    x = np.vstack((ones,x))
    hypothesis = np.matmul(theta.T,x)
    loss = np.power((hypothesis-y),2)
    return np.sum(loss)

def df2(x,y,theta):
    ones = np.ones((1,x.shape[1]))
    x = vstack((ones,x))
    hypothesis = np.matmul(np.transpose(theta),x)-y
    hypothesis = np.transpose(hypothesis)
    gradient = 2*np.matmul(x,hypothesis)
    return gradient


def label(x):
    label = flattenData(x)
    return np.sum(label)

def getDataMatrix(act, partNumber):
    directory = str("cropped" + str(partNumber) + "/")
    directory = os.path.join(os.getcwd(),directory)
    x = []
    for a in act:
        name = a.split()[1].lower()
        for file in os.listdir(directory):
            filename = str(file)
            if name in filename:
                img = imread(os.path.join(directory,str(file)))
                img = np.array(img)/255.0
                img = img.flatten()
                x.append(img)
    x = np.array(x)
    return x

def getDataLabels(act,partNumber):
    partNo = str("cropped" + str(partNumber)+ "/")
    i=0.0
    directory = os.path.join(os.getcwd(),partNo)
    y = []
    for a in act:
        name = a.split()[1].lower()
        print(name)
        for file in os.listdir(directory):
            filename = str(file)
            if name in filename:
                y.append(i)
        i+=1.0
    return np.transpose(y)

def getMultipleDataLabels(act, partNumber):
    partNo = str("cropped" + str(partNumber)+"/")
    i = 0
    directory = os.path.join(os.getcwd(),partNo)
    names = []
    for a in act:
        name = a.split()[1].lower()
        names.append(name)
        for file in os.listdir(directory):
            filename = str(file)
            if name in filename:
                i+=1
    k = len(names)
    y = np.zeros([k,i])
    i = 0
    k = 0
    for a in act:
        name = a.split()[1].lower()
        for file in os.listdir(directory):
            filename = str(file)
            if name in filename:
                y[k,i]=1
                i+=1
        k+=1
    return y

def getFileList(directory):
    for file in os.listdir(directory):
        print(str(file))

def part3():
    #############################################################################################################
    '''
    Code for Part 3 of the assignment - proceeds by getting cropped images of both Alec Baldwin and Steve Carell.
    '''
    #############################################################################################################
    
    #Declare list of actors for processing
    act = ['Alec Baldwin', 'Steve Carell']
    getCroppedData(act,"facescrub_actors.txt",3)
    x = getDataMatrix(act,3)
    y = getDataLabels(act,3)
    print(x.shape)
    #concatenate the data matrix and labels for processing
    
    complete = np.column_stack((x,y))
    np.random.seed(4)
    np.random.shuffle(complete)
    training = complete[0:200,:]
    validation = complete[200:230,:]
    test = complete[230:260,:]
    print("Number of Baldwin in training set: " + str((np.shape(np.where(training[:,-1]==0))[1])))
    print("Number of Baldwin in validation set: " + str((np.shape(np.where(validation[:,-1]==0))[1])))
    print("Number of Baldwin in test set: " + str((np.shape(np.where(test[:,-1]==0))[1])))
    print("Number of Carell in training set: " + str((np.shape(np.where(training[:,-1]==1))[1])))
    print("Number of Carell in validation set: " + str((np.shape(np.where(validation[:,-1]==1))[1])))
    print("Number of Carell in test set: " + str((np.shape(np.where(test[:,-1]==1))[1])))
    theta0 = np.ones((1025,))

    training_labels = training[:,-1]
    training = np.transpose(training[:,:-1])
    print(training.shape)
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,10000)

    #Training hypothesis
    ones_t = np.ones((1,training.shape[1]))
    training_with_bias = vstack((ones_t,training))
    training_hypothesis = np.dot(theta.T,training_with_bias)
    i = 0
    while i < training_hypothesis.shape[0]:
        if training_hypothesis[i] > 0.5:
            training_hypothesis[i] = 1
        else:
            training_hypothesis[i] = 0
        i+=1
    print("Accuracy percentage in training set:" + str(np.sum(np.equal(training_hypothesis,training_labels))/200.0))

    #Validation hypothesis
    validation_labels = validation[:,-1]
    validation = np.transpose(validation[:,:-1])
    ones_v = np.ones((1,validation.shape[1]))
    validation_with_bias = vstack((ones_v,validation))
    validation_hypothesis = np.dot(theta.T,validation_with_bias)
    i=0
    while i < validation_hypothesis.shape[0]:
        if validation_hypothesis[i] > 0.5:
            validation_hypothesis[i] = 1
        else:
            validation_hypothesis[i] = 0
        i+=1

    print("Accuracy percentage in validation set:" + str(np.sum(np.equal(validation_hypothesis,validation_labels))/30.0))

    test_labels = test[:,-1]
    test = np.transpose(test[:,:-1])
    ones_test = np.ones((1,test.shape[1]))
    test_with_bias = vstack((ones_test,test))
    test_hypothesis = np.dot(theta.T,test_with_bias)
    i=0
    while i < test_hypothesis.shape[0]:
        if test_hypothesis[i] > 0.5:
            test_hypothesis[i] = 1
        else:
            test_hypothesis[i] = 0
        i+=1

    print("Accuracy percentage in test set:" + str(np.sum(np.equal(test_hypothesis,test_labels))/30.0))
    return theta

def part4a(theta):
    '''
    #####################################################################################################################
    Code for Part 4
    #####################################################################################################################
    '''
    #Outputting thetas for full training set of 100 for both Baldwin and Carell
    theta = theta[1:]
    print(theta.shape)
    test = np.reshape(theta,(32,32))
    plt.imsave("complete_thetas.jpg",test,cmap='RdBu')
    
    
    

    #Outputting thetas for training set of 2 images from both sets
    #Declare list of actors for processing
    act = ['Alec Baldwin', 'Steve Carell']
    #getCroppedData(act,"facescrub_actors.txt",3)
    x = getDataMatrix(act,3)
    print(x.shape)
    y = getDataLabels(act,3)
    complete = np.column_stack((x,y))

    #Get two from each
    training0 = complete[random.randint(0,130),:]
    training1 = complete[random.randint(0,130),:]
    training2 = complete[random.randint(131,260),:]
    training3 = complete[random.randint(131,260),:]
    while np.array_equal(training0,training1):
        training1 = complete[random.randint(0,130),:]
    while np.array_equal(training2,training3):
        training3 = complete[random.randint(131,260),:]
    small_train = vstack((training0,training1))
    small_train = vstack((small_train,training2))
    small_train = vstack((small_train,training3))
    small_train_labels = small_train[:,-1]
    small_train = np.transpose(small_train[:,:-1])

    #Saving thetas for two actors
    theta0 = np.ones((1025,))
    theta = grad_descent(f,df,small_train,small_train_labels,theta0,0.0001,10000)
    theta = theta[1:]
    print(theta.shape)
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas.jpg',test,cmap='RdBu')
    
    

    #Implementing gradient descent for thetas on full training set by modifying the number of iterations and initializing theta differently
    training = complete[0:200,:]
    print("Number of Baldwin in training set: " + str((np.shape(np.where(training[:,-1]==0))[1])))
    print("Number of Carell in training set: " + str((np.shape(np.where(training[:,-1]==1))[1])))
    training_labels = training[:,-1]
    training = np.transpose(training[:,:-1])
    theta0 = np.ones((1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,10000)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas_ones_10000_iterations.jpg',test,cmap='RdBu')
    
    theta0 = np.random.normal(0,0.2,(1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,10000)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas__norm_10000_iterations.jpg',test,cmap='RdBu')

    theta0 = np.random.lognormal(0,0.2,(1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,10000)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas__lognorm_10000_iterations.jpg',test,cmap='RdBu')

    theta0 = np.ones((1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,7500)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas_ones_7500_iterations.jpg',test,cmap='RdBu')
    
    theta0 = np.random.normal(0,0.2,(1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,7500)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas__norm_7500_iterations.jpg',test,cmap='RdBu')

    theta0 = np.random.lognormal(0,0.2,(1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,7500)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas__lognorm_7500_iterations.jpg',test,cmap='RdBu')

    theta0 = np.ones((1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,5000)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas_ones_5000_iterations.jpg',test, cmap='RdBu')
    
    theta0 = np.random.normal(0,0.2,(1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,5000)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas__norm_5000_iterations.jpg',test,cmap='RdBu')

    theta0 = np.random.lognormal(0,0.2,(1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,5000)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas__lognorm_5000_iterations.jpg',test,cmap='RdBu')

    theta0 = np.ones((1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,2500)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas_ones_2500_iterations.jpg',test, cmap='RdBu')
    
    theta0 = np.random.normal(0,0.2,(1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,2500)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas__norm_2500_iterations.jpg',test,cmap='RdBu')

    theta0 = np.random.lognormal(0,0.2,(1025,))
    theta = grad_descent(f,df,training,training_labels,theta0,0.00001,2500)
    theta = theta[1:]
    test = np.reshape(theta,(32,32))
    plt.imsave('two_actor_thetas__lognorm_2500_iterations.jpg',test,cmap='RdBu')


    '''
    theta = grad_descent(f,df,small_train,small_train_labels,theta0,0.001,10000)
    theta = theta[1:]
    print(theta.shape)
    test = np.reshape(theta,(32,32))
    imsave('two_actor_thetas_learning_rate0.jpg',cmap='RdBu')
    print(test.shape)
    plt.imshow(test,cmap='RdBu')
    plt.show()

    theta0 = np.ones((1025,))
    theta = grad_descent(f,df,small_train,small_train_labels,theta0,0.001,10000)
    theta = theta[1:]
    print(theta.shape)
    test = np.reshape(theta,(32,32))
    imsave('two_actor_thetas_learning_rate0.jpg',cmap='RdBu')
    print(test.shape)
    plt.imshow(test,cmap='RdBu')
    plt.show()
    '''

def part5():
    
    #############################################################################################################
    '''
    Code for Part 5 of the assignment - proceeds by getting cropped images of both Alec Baldwin and Steve Carell.
    '''
    #############################################################################################################
        
    act =['Alec Baldwin', 'Bill Hader', 'Steve Carell']
    getCroppedData(act,"facescrub_actors.txt",5)
    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
    getCroppedData(act, "facescrub_actresses.txt",5)
    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    x = getDataMatrix(act, 5)
    y = getDataLabels(act,5)
    i = 0
    #Cleaning up data matrix for classification - 0 is female, 1 is male
    while i < y.shape[0]:
        if y[i] < 3:
            y[i] = 0
        else:
            y[i] = 1
        i+=1
    
    complete = np.column_stack((x,y))
    np.random.seed(1)
    np.random.shuffle(complete)

    i = 50
    while i < 601:
        #Using the full set of training data
        training = complete[0:i,:]
        validation = complete[i+1:i+int(i*0.8),:]
        print("Number of Females in training set: " + str((np.shape(np.where(training[:,-1]==0))[1])))
        print("Number of Males in training set: " + str((np.shape(np.where(training[:,-1]==1))[1])))
        print("Number of Females in validation set: " + str((np.shape(np.where(validation[:,-1]==0))[1])))
        print("Number of Males in validation set: " + str((np.shape(np.where(validation[:,-1]==1))[1])))
        
        theta0 = np.ones((1025,))
        training_labels = training[:,-1]
        training = np.transpose(training[:,:-1])
        theta1 = grad_descent(f,df,training,training_labels,theta0,0.000001,10000)

        #Training hypothesis
        ones_t = np.ones((1,training.shape[1]))
        training_with_bias = vstack((ones_t,training))
        training_hypothesis = np.dot(theta1.T,training_with_bias)
        j = 0
        while j < training_hypothesis.shape[0]:
            if training_hypothesis[j] > 0.5:
                training_hypothesis[j] = 1
            else:
                training_hypothesis[j] = 0
            j+=1
    
        print("Accuracy percentage in training set:" + str(np.sum(np.equal(training_hypothesis,training_labels))/float(i)))
    
        validation_labels = validation[:,-1]
        validation = np.transpose(validation[:,:-1])
        ones_v = np.ones((1,validation.shape[1]))
        validation_with_bias = vstack((ones_v,validation))
        validation_hypothesis = np.dot(theta1.T,validation_with_bias)
        j = 0
        while j < validation_hypothesis.shape[0]:
            if validation_hypothesis[j] > 0.5:
                validation_hypothesis[j] = 1
            else:
                validation_hypothesis[j] = 0
            j+=1

        print("Accuracy percentage in validation set: " + str(np.sum(np.equal(validation,validation_labels))/(float(i*0.8))))
        i+=50

def part6():
    #Testing the loss function
    x = np.random.normal(0,0.6,(20,15))
    y = np.random.normal(0.2,0.4,(5,15))
    theta = np.random.normal(-0.1,0.3,(21,5))
    h = 0.00001
    #Cost of individual component
    testarr1 = np.zeros(theta.shape)
    testarr1[3,4] = h
    print("Cost is:")
    print((f2(x,y,theta+testarr1)-f2(x,y,theta-testarr1))/(2*h))
    print("Gradient is:")
    print(df2(x,y,theta)[3,4])
    testarr2 = np.zeros(theta.shape)
    testarr2[1,4] = h
    print("Cost is:")
    print((f2(x,y,theta+testarr2)-f2(x,y,theta-testarr2))/(2*h))
    print("Gradient is:")
    print(df2(x,y,theta)[1,4])
    testarr3 = np.zeros(theta.shape)
    testarr3[2,3] = h
    print("Cost is:")
    print((f2(x,y,theta+testarr3)-f2(x,y,theta-testarr3))/(2*h))
    print("Gradient is:")
    print(df2(x,y,theta)[2,3])
    testarr4 = np.zeros(theta.shape)
    testarr4[8,4] = h
    print("Cost is:")
    print((f2(x,y,theta+testarr4)-f2(x,y,theta-testarr4))/(2*h))
    print("Gradient is:")
    print(df2(x,y,theta)[8,4])
    testarr5 = np.zeros(theta.shape)
    testarr5[10,1] = h
    print("Cost is:")
    print((f2(x,y,theta+testarr5)-f2(x,y,theta-testarr5))/(2*h))
    print("Gradient is:")
    print(df2(x,y,theta)[10,1])
    


def part7():
    #Rough attempt at classification
    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    x = getDataMatrix(act, 5)
    x = np.transpose(x)
    y = getMultipleDataLabels(act,5)
    print("Output guess shape:")
    print(y.shape)
    complete = np.vstack((x,y))
    
    np.random.shuffle(complete.T)
    training = complete[:,0:600]
    print("After shuffling complete data matrix:")
    print(training.shape)
    validation = complete[:,601:691]
    theta0 = np.random.normal(0, 0.2,(1025,y.shape[0]))
    training_labels = complete[-y.shape[0]:, 0:600]
    print("Training label size:")
    print(training_labels.shape)
    training = training[:-training_labels.shape[0]]
    print("After splicing")
    print(training.shape)
    theta = grad_descent(f2,df2,training,training_labels,theta0,0.000005,10000)

    ones_t = np.ones((1,training.shape[1]))
    training_with_bias = vstack((ones_t,training))
    training_hypothesis = np.matmul(theta.T,training_with_bias)
    max = training_hypothesis.argmax(axis=0)
    max = np.array(max)
    training_hypothesis_labels = np.zeros((training_hypothesis.shape))
    print(training_hypothesis_labels.shape)
    i = 0
    while i < training_hypothesis_labels.shape[1]:
        index = max[i]
        training_hypothesis_labels[index,i]=1
        i+=1

    print(training_hypothesis_labels)
    print(training_labels)
    '''
    correct = 0
    i = 0
        
    while i < training_hypothesis_labels.shape[1]:
        if np.array_equal(training_hypothesis[:,i],training_labels[:,i]) == True:
            correct +=1
        i+=1
    print(correct)
    '''
    
    '''
    complete = np.column_stack((x,y))
    np.random.shuffle(complete)
    training = complete[:,200]
    validation = complete[200:230,:]
    test = complete[230:260,:]
    training = np.transpose(complete[:,:-y.shape[1]])
    print(training.shape)
    training_labels = complete[:,-y.shape[1]:]
    theta0 = np.ones((1025,y.shape[1]))
       
    #Training thetas:
    theta = grad_descent(f2,df2,training,training_labels,theta0,0.00001,1)

    ones_t = np.ones((1,training.shape[1]))
    training_with_bias = vstack((ones_t,training))
    training_hypothesis = np.matmul(theta.T,training_with_bias)
    max = training_hypothesis.argmax(axis=0)
    max = np.array(max)
    new = np.zeros((y.shape[1],training.shape[1]))
    i = 0
    while i < new.shape[1]:
        index = max[i]
        new[index,i]=1
        i+=1
    training_hypothesis = new
    print(training_hypothesis)
    print(np.sum(np.equal(training_hypothesis,training_labels.T)))
    '''

    
    
    
    '''
    #Validation hypothesis
    validation_labels = validation[:,-1]
    validation = np.transpose(validation[:,:-1])
    ones_v = np.ones((1,validation.shape[1]))
    validation_with_bias = vstack((ones_v,validation))
    validation_hypothesis = np.dot(theta.T,validation_with_bias)
    i=0
    while i < validation_hypothesis.shape[0]:
        if validation_hypothesis[i] > 0.5:
            validation_hypothesis[i] = 1
        else:
            validation_hypothesis[i] = 0
        i+=1

    print("Accuracy percentage in validation set:" + str(np.sum(np.equal(validation_hypothesis,validation_labels))/30.0))
    '''


    
def main():
    #theta = part3()
    #part4a(theta)
    #part5()
    #part6()
    part7()



if __name__ == "__main__":
    main()