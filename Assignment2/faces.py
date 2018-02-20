import os
import hashlib
import ssl
import socket
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


# Imported code from Assignment 1 to handle FaceScrub data set

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

def getPictures():

    location = str("cropped/")
    if not os.path.exists(os.path.join(os.getcwd(), location)):
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
                        sha_hash = str(line.split()[6])
                        print("Attempting to look at URL:" + url)
                        print("Filename is: " + filename)
                        print("Getting the bounding box...")
                        x1 = int(line.split()[5].split(",")[0])
                        y1 = int(line.split()[5].split(",")[1])
                        x2 = int(line.split()[5].split(",")[2])
                        y2 = int(line.split()[5].split(",")[3])
                        print("Checking hash...")
                        img = urllib.request.urlopen(url, timeout=5)
                        m = hashlib.sha256(img.read()).hexdigest()
                        if str(m) == sha_hash:
                            print("Hash matches, checking for RGB...")
                            try:
                                img = imread(urllib.request.urlopen(url, timeout=5))
                                if len(img.shape) < 3:
                                    print("Image is not colour, skipping....")
                                else:
                                    print("Hash matched, colour image, saving to disk...")
                                    img = img[y1:y2, x1:x2]
                                    img = img/255.0
                                    img = imresize(img, [32, 32], 'nearest')
                                    print("Attempting to save file to disk")
                                    imsave(os.path.join(os.path.join(os.getcwd(), location), filename), img)
                                    i += 1
                            except IOError as e:
                                print("Unable to read image")
                            except ValueError as e:
                                print("Array corrupted")
                        else:
                            print("Hash does not match...")
                    except urllib.error.HTTPError as e:
                        print("HTTP Error")
                    except urllib.error.URLError as e:
                        print("URL Error")
                    except urllib.error.ContentTooShortError as e:
                        print("Content too short")
                    except Exception:
                        print("Other exception")

def getDataMatrix(actors):
    directory = os.path.join(os.getcwd(), "cropped/")
    x = np.zeros((32 * 32 * 3, 0))
    for a in actors:
        name = a.split()[1].lower()
        for file in os.listdir(directory):
            filename = str(file)
            if name in filename:
                img = imread(os.path.join(directory, str(file)))
                img = np.array(img)/255.0
                img = img.flatten()
                img = np.expand_dims(img, axis=1)
                x = np.hstack((x, img))
    x = np.array(x)
    return x


def getDataLabels(actors):
    directory = os.path.join(os.getcwd(), "cropped/")
    batch_y_s = np.zeros((0, 6))
    current_index = 0
    for a in actors:
        name = a.split()[1].lower()
        for file in os.listdir(directory):
            filename = str(file)
            if name in filename:
                one_hot = np.zeros(6)
                one_hot[current_index] = 1
                batch_y_s = np.vstack((batch_y_s, one_hot))
        current_index += 1

    return np.transpose(batch_y_s)


def shuffle(data, labels, percentage):
    np.random.seed(1)
    complete = np.vstack((data, labels))
    complete = np.transpose(complete)
    np.random.shuffle(complete)
    complete = np.transpose(complete)
    trainData, validData = complete[:-6, :int(percentage*complete.shape[1])], complete[:-6, int(percentage*complete.shape[1]):int(complete.shape[1])]
    trainLabels, validLabels = complete[3072:, :int(percentage*complete.shape[1])], complete[3072:, int(percentage*complete.shape[1]):int(complete.shape[1])]
    return trainData, validData, trainLabels, validLabels


# Second main function


def main():

    # Declaring list of actors
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    # Getting the data from the URLs
    getPictures()
    X = getDataMatrix(act)
    Y = getDataLabels(act)
    trainData, validData, trainLabels, validLabels = shuffle(X, Y, 0.8)
    trainData = np.transpose(trainData)
    trainLabels = np.transpose(trainLabels)
    validData = np.transpose(validData)
    validLabels = np.transpose(validLabels)
    print(trainData.shape)
    print(validData.shape)
    print(trainLabels.shape)
    print(validLabels.shape)

    # Setting up the pytorch model
    dim_x = 32*32*3
    dim_h = 20
    dim_out = 6
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h), torch.nn.ReLU(), torch.nn.Linear(dim_h, dim_out),)
    loss_func = torch.nn.CrossEntropyLoss()

    x = Variable(torch.from_numpy(trainData), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(trainLabels, 1)), requires_grad=False).type(dtype_long)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(10000):
        y_pred = model(x)
        loss = loss_func(y_pred, y_classes)
        model.zero_grad()
        loss.backward()
        optimizer.step()

    x = Variable(torch.from_numpy(validData), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()
    print(np.mean(np.argmax(y_pred, 1)==np.argmax(validLabels, 1)))

    # filename = os.path.join(os.getcwd(), "cropped/")
    # filename = filename + str("baldwin0.jpg")
    # test = imread(filename)
    # test = imresize(test, [32,32], 'nearest')
    # plt.imshow(test)
    # plt.show()

if __name__ == "__main__":
    main()
