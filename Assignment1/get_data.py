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

def getCroppedData(list,file):
    if not os.path.exists(os.path.join(os.getcwd(),"cropped/")):
        os.makedirs(os.path.join(os.getcwd(),"cropped/"))
    for a in list:
        name = a.split()[1].lower()
        i=0
        for line in open(getActors(file,1)):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                if filename.endswith(".jpg") and i != 130:
                    try:
                        if os.path.isfile(os.path.join(os.path.join(os.getcwd(),"cropped"),filename)):
                            continue
                        url = line.split()[4]
                        print("Attempting to look at URL:" + url)
                        
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
                            imsave(os.path.join(os.path.join(os.getcwd(),"cropped"),filename),img)
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
    
def grad_descent(f,df,x,y,init_t,alpha):
    EPS = 1e-5
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 10000
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

def label(x):
    label = flattenData(x)
    return np.sum(label)

def getDataMatrix():
    directory = os.path.join(os.getcwd(),"cropped")
    x = []
    for file in os.listdir(directory):
        img = imread(os.path.join(directory,str(file)))
        img = np.array(img)/255.0
        img = img.flatten()
        x.append(img)
    x = np.array(x)
    return x

def getDataLabels(act):
    i=0.0
    directory = os.path.join(os.getcwd(),"cropped")
    y = []
    for a in act:
        name = a.split()[1].lower()
        for file in os.listdir(directory):
            filename = str(file)
            if name in filename:
                y.append(i)
        i+=1.0
    return np.transpose(y)



#Declare list of actors for processing
act = ['Alec Baldwin', 'Steve Carell']
#getCroppedData(act,"facescrub_actors.txt")
x = getDataMatrix()
print(x.shape)
y = getDataLabels(act)
print(y.shape)


#concatenate the data matrix and labels for processing
print(x.shape)
print(y.shape)
complete = np.column_stack((x,y))
print(complete.shape)
np.random.seed(1)
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
theta0 = np.random.normal(0,0.2,1025)

training_labels = training[:,-1]
training = np.transpose(training[:,:-1])
theta = grad_descent(f,df,training,training_labels,theta0,0.00001)

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

theta = theta[1:]
theta = np.reshape(theta,(32,32))
plt.imshow(theta)
plt.show()


'''
np.random.shuffle(complete)
training = complete[0:200,:]
validation = complete[200:220,:]
test = complete[220:240:]
print("Number of Baldwin in training set: " + str((np.shape(np.where(training[:,-1]==0))[1])))
print("Number of Baldwin in validation set: " + str((np.shape(np.where(validation[:,-1]==0))[1])))
print("Number of Baldwin in test set: " + str((np.shape(np.where(test[:,-1]==0))[1])))
print("Number of Carell in training set: " + str((np.shape(np.where(training[:,-1]==1))[1])))
print("Number of Carell in validation set: " + str((np.shape(np.where(validation[:,-1]==1))[1])))
print("Number of Carell in test set: " + str((np.shape(np.where(test[:,-1]==1))[1])))
theta0 = np.random.normal(0.1,0.05,1025)


training = training[:,:-1]
print(training.shape)
print(np.transpose(training).shape)
ones = np.ones((1,np.shape(training)[0]))
print(ones.shape)
sos = np.vstack((ones,np.transpose(training)))
print(sos.shape)

theta = grad_descent(f,df,np.transpose(training[:,:-1]),training[:,-1],theta0,0.00001)

#Generating a hypothesis on training set:
hypothesis = np.matmul(theta.T, np.vstack((np.ones((1,np.shape(training[:,:-1])[0])),np.transpose(training[:,:-1]))))
#Report on accuracy
i = 0
while i < hypothesis.shape[0]:
    if hypothesis[i] > 0.5:
        hypothesis[i] = 1
    else:
        hypothesis[i] = 0
    i+=1

trainingLabels = training[:,-1]
print("Training accuracy is: " + str(sum(np.equal(hypothesis,trainingLabels))/200.0))

validation_hypothesis=np.matmul(theta.T, np.vstack((np.ones((1,np.shape(validation[:,:-1])[0])),np.transpose(validation[:,:-1]))))
i=0
while i < validation_hypothesis.shape[0]:
    if validation_hypothesis[i] > 0.5:
        validation_hypothesis[i] = 1
    else:
        validation_hypothesis[i] = 0
    i+=1

validationLabels = test[:,-1]
print("Validation accuracy is: " + str(sum(np.equal(validation_hypothesis,validationLabels))/40.0))


test_hypothesis = np.matmul(theta.T, np.vstack((np.ones((1,np.shape(test[:,:-1])[0])),np.transpose(test[:,:-1]))))
i=0
while i < test_hypothesis.shape[0]:
    if test_hypothesis[i] > 0.5:
        test_hypothesis[i] = 1
    else:
        test_hypothesis[i] = 0
    i+=1

testingLabels = test[:,-1]
print("Testing accuracy is: " + str(sum(np.equal(test_hypothesis,testingLabels))/40.0))
theta = theta[1:]
theta = np.reshape(theta,(32,32))
plt.imshow(theta)
plt.show()
'''