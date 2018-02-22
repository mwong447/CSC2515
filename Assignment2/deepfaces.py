import os
import hashlib
from scipy.misc import imread
from scipy.misc import imshow
from scipy.misc import imresize
from scipy.misc import imsave
from torch.autograd import Variable
import urllib.error
import urllib.request
import urllib.parse
import urllib.response
import urllib.robotparser
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torchvision.models as models


# Functions from faces.py for this section
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

    location = str("cropped227/")
    if not os.path.exists(os.path.join(os.getcwd(), location)):
        males = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']
        getCroppedData(males, "facescrub_actors.txt")
        females = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
        getCroppedData(females, "facescrub_actresses.txt")

def getCroppedData(list, file):
    location = str("cropped227/")
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
                                    img = imresize(img, [227, 227], 'nearest')
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
    directory = os.path.join(os.getcwd(), "cropped227/")
    x = np.zeros((0, 3, 227, 227))
    for a in actors:
        name = a.split()[1].lower()
        for file in os.listdir(directory):
            filename = str(file)
            if name in filename:
                img = imread(os.path.join(directory, str(file)))[:, :, :3]
                img = img - np.mean(img.flatten())
                img = img/255.0
                img = np.rollaxis(img, -1).astype("float32")
                img = img[np.newaxis, :]
                x = np.concatenate((x, img))
    return x


def getDataLabels(actors):
    directory = os.path.join(os.getcwd(), "cropped227/")
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

    # Getting the test data
    complete = np.vstack((data, labels))
    bracco = int(np.sum(complete[154587, :], axis=0))
    gilpin = int(np.sum(complete[154588, :], axis=0))
    harmon = int(np.sum(complete[154589, :], axis=0))
    baldwin = int(np.sum(complete[154590, :], axis=0))
    hader = int(np.sum(complete[154591, :], axis=0))
    carell = int(np.sum(complete[154592, :], axis=0))

    # creating actor-specific subarrays and extracting the first twenty examples
    bracco_matrix = complete[:, :bracco]
    gilpin_matrix = complete[:, bracco:bracco+gilpin]
    harmon_matrix = complete[:, bracco+gilpin:bracco+gilpin+harmon]
    baldwin_matrix = complete[:, bracco+gilpin+harmon:bracco+gilpin+harmon+baldwin]
    hader_matrix = complete[:, bracco+gilpin+harmon+baldwin:bracco+gilpin+harmon+baldwin+hader]
    carell_matrix = complete[:, bracco+gilpin+harmon+baldwin+hader:bracco+gilpin+harmon+baldwin+hader+carell]
    test_bracco = bracco_matrix[:, :20]
    test_gilpin = gilpin_matrix[:, :20]
    test_harmon = harmon_matrix[:, :20]
    test_baldwin = baldwin_matrix[:, :20]
    test_hader = hader_matrix[:, :20]
    test_carell = carell_matrix[:, :20]
    testData = np.hstack((test_bracco, test_gilpin))
    testData = np.hstack((testData, test_harmon))
    testData = np.hstack((testData, test_baldwin))
    testData = np.hstack((testData, test_hader))
    testData = np.hstack((testData, test_carell))
    testLabels = testData[154587:, :]
    testData = testData[:-6, :]


    # Need to do something similar for the remaining dataset

    bracco_matrix = bracco_matrix[:, 20:]
    gilpin_matrix = gilpin_matrix[:, 20:]
    harmon_matrix = harmon_matrix[:, 20:]
    baldwin_matrix = baldwin_matrix[:, 20:]
    hader_matrix = hader_matrix[:, 20:]
    carell_matrix = carell_matrix[:, 20:]
    complete = np.hstack((bracco_matrix, gilpin_matrix))
    complete = np.hstack((complete, harmon_matrix))
    complete = np.hstack((complete, baldwin_matrix))
    complete = np.hstack((complete, hader_matrix))
    complete = np.hstack((complete, carell_matrix))
    complete = np.transpose(complete)
    np.random.shuffle(complete)
    complete = np.transpose(complete)
    trainData, validData = complete[:-6, :int(percentage*complete.shape[1])], complete[:-6, int(percentage*complete.shape[1]):int(complete.shape[1])]
    trainLabels, validLabels = complete[154587:, :int(percentage*complete.shape[1])], complete[154587:, int(percentage*complete.shape[1]):int(complete.shape[1])]

    return trainData, validData, testData, trainLabels, validLabels, testLabels



class AnotherAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)

        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

        classifier_weight_i = [1, 4]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(AnotherAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        return x


def main():

    model = AnotherAlexNet()
    model.eval()
    # Declaring list of actors
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    # Getting the data from the URLs
    getPictures()
    X = getDataMatrix(act)

    softmax = torch.nn.Softmax(dim=1)
    testvar = Variable(torch.from_numpy(X), requires_grad=False)
    print(testvar.shape)
    all_probs = softmax(model.forward(testvar)).data.numpy()
    print(all_probs.shape)


    # directory = os.path.join(os.getcwd(),"cropped227/")
    # im = imread(os.path.join(directory,"baldwin0.jpg"))[:,:,:3]
    # im = im - np.mean(im.flatten())
    # im = im / np.max(np.abs(im.flatten()))
    # im = np.rollaxis(im, -1).astype("float32")
    #
    # im2 = imread(os.path.join(directory, "baldwin1.jpg"))
    # im2 = im2 - np.mean(im2.flatten())
    # im2 = im2 / np.max(np.abs(im2.flatten()))
    # im2 = np.rollaxis(im2, -1).astype("float32")
    #
    # softmax = torch.nn.Softmax()
    # soosh = Variable(torch.from_numpy(im2).unsqueeze_(0), requires_grad=False)
    # print(soosh.shape)
    # testoutput = softmax(model1.forward(soosh)).data.numpy()
    # print(testoutput.shape)







if __name__ == "__main__":
        main()


