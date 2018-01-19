from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import socket



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

def getCroppedData():
    actors = "facescrub_actors.txt"
    list = ['Alec Baldwin', 'Steve Carell']
    #list = getActors(actors)
    if not os.path.exists(os.path.join(os.getcwd(),"cropped/")):
        os.makedirs(os.path.join(os.getcwd(),"cropped/"))
    for a in list:
        name = a.split()[1].lower()
        i=0
        for line in open(getActors(actors,1)):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                if filename.endswith(".jpg"):
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

#getCroppedData()
loc = os.path.join(os.getcwd(),"cropped/baldwin0.jpg")
img = imread(loc)
img = flattenData(img)
print(img)

'''
#Test program to crop
url = "http://images2.fanpop.com/image/photos/12400000/Gerry-gerard-butler-12419263-1617-2000.jpg"
img = imread(urllib.request.urlopen(url))
img = img[420:1479,397:1456]
img = img/255.0
plt.imshow(img)
plt.show()
#print(img.shape)
#img = rgb2gray(img)
#img = imresize(img,[32,32],'nearest')
#plt.imshow(img)
#plt.show()
'''









'''
for a in list:
    name = a.split()[1].lower()
    i = 0
    for line in open(getActors(actors,1)):
        filename = name + str(i)+'.'+line.split()[4].split('.')[-1]
        timeout(urllib.request.urlopen,(line.split()[4],"uncropped/"+filename),{},30)
        if not os.path.isfile("uncropped/"+filename):
            continue
        print(filename)
        i+=1
'''