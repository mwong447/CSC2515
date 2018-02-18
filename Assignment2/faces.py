import os
from scipy.misc import imread
from scipy.misc import imshow
from scipy.misc import imresize
from scipy.misc import imsave
import urllib.error
import urllib.request
import urllib.parse
import urllib.response
import urllib.robotparser
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


#Imported code from Assignment 1 to handle FaceScrub dataset

def getActors(X,Y=None):
    current_directory = os.getcwd()
    filename = str(X)
    filename = os.path.join(current_directory,X)
    if Y is None:
        act = list(set([a.split("\t")[0] for a in open(filename).readlines()]))
        print(act)
        return act
    else:
        return filename

def getpictures():

    location = str("cropped/")
    if not os.path.exists(os.path.join(os.getcwd(),location)):
        males = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']
        getCroppedData(males, "facescrub_actors.txt")
        females = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
        getCroppedData(females, "facescrub_actresses.txt")


def getCroppedData(list, file):
    location = str("cropped/")
    if not os.path.exists(os.path.join(os.getcwd(), location)):
        os.makedirs(os.path.join(os.getcwd(), location))
    for a in list:
        name = a.split()[1].lower()
        i=0
        for line in open(getActors(file, 1)):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                if filename.endswith(".jpg") and i != 130:
                    try:
                        if os.path.isfile(os.path.join(os.path.join(os.getcwd(), location), filename)):
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
                            img = imread(urllib.request.urlopen(url, timeout=5))
                            img = img[y1:y2,x1:x2]
                            img = img/255.0
                            img = imresize(img,[32,32],'nearest')
                            print("Attempting to save file to disk")
                            imsave(os.path.join(os.path.join(os.getcwd(), location), filename), img)
                            i+=1
                        except IOError:
                            print("Unable to read image")
                        except ValueError:
                            print("Array corrupted")
                    except urllib.error.HTTPError as e:
                        print("Unable to get content")
                    except urllib.error.URLError as e:
                        print("Unable to reach URL")
                    except urllib.error.ContentTooShortError as e:
                        print("Content too short")


#Second main function


def main():

    getpictures()
    filename = os.path.join(os.getcwd(), "cropped/")
    filename = filename + str("baldwin0.jpg")
    test = imread(filename)
    test = imresize(test, [32,32], 'nearest')
    plt.imshow(test)
    plt.show()

if __name__ == "__main__":
    main()
