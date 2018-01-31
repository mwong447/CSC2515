from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np



x = np.random.normal(0,100,(1024))
y1 = np.zeros((512))
y2 = np.ones((512))
y = np.hstack((y1,y2))
complete = np.vstack((x,y))
complete = np.transpose(complete)
np.random.seed(1)
np.random.shuffle(complete)
labels = complete[:,1]
complete = complete[:,:-1]
i = 1
while i < 1000:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(complete, labels)
    y_pred = knn.predict(complete)
    print("K: " + str(i))
    print(str((accuracy_score(labels,y_pred))))
    i+=1

    
