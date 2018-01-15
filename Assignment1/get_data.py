from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
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
        return act
    else:
        return FileName

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default
    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result



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
                   page = urllib.request.urlretrieve(line.split()[4],os.path.join(os.path.join(os.getcwd(),"uncropped"),filename))
                   print("Attempting to write to disk: " + filename)
                   i+=1
                except urllib.error.HTTPError:
                    print("Unable to get content")
                except urllib.error.URLError:
                    print("Unable to reach URL")
                except urllib.error.ContentTooShortError:
                    print("Content too short")


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