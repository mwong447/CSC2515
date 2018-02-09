from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
#Set up seed and datasets
np.random.seed(100)
x1 = np.random.normal(loc = 1.0, scale = 0.15,size=(25,2))
x2 = np.random.normal(loc = 1.0, scale = 5.0,size=(75,2))
x3 = np.random.normal(loc = 6.0, scale = 0.15,size = (10,2))
#Plot and save the data
plt.scatter(x1[:,0],x1[:,1],label= "0 labels")
plt.scatter(x2[:,0],x2[:,1],label = "1 labels")
plt.scatter(x3[:,0],x3[:,1],label = "0 labels")
plt.legend()
plt.savefig('Data_plot.jpg')
plt.close()
#Stacking the matrices for X
x = np.vstack((x1,x2))
x = np.vstack((x,x3))
#Generate corresponding labels for clusters
y1 = np.zeros((25))
y2 = np.ones((75))
y3 = np.zeros((10))
#Stacking the labels
y = np.hstack((y1,y2))
y = np.hstack((y,y3))
#Stacking the matrices together
y = y[:,np.newaxis]
complete = np.hstack((x,y))
#Shuffle the matrix
np.random.shuffle(complete)
#Get the labels
labels = complete[:,-1]
#Get the data
complete = complete[:,:-1]
#KNN implementation
k_values, accuracy = [],[]
i = 1
while i < complete.shape[0]:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(complete, labels)
    y_pred = knn.predict(complete)
    k_values.append(i)
    accuracy.append((accuracy_score(labels,y_pred)))
    i+=2
#Plotting the data
x_axis = k_values
y_axis = accuracy
plt.xlabel("Number of Nearest Neighbours")
plt.ylabel("Percentage Accuracy")
plt.plot(x_axis, y_axis, 'b--')
plt.savefig('Performance_Chart.jpg')