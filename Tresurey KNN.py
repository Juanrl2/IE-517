import pandas as pd
import sklearn
import numpy as np
print( 'The scikit learn version is {}.'.format(sklearn.__version__))
path = "D:\Machine Learning\TSRD(1).csv"
treasury = pd.read_csv(path)
treasury = treasury.values
treasury = treasury[:,2:]
X, y = treasury[:,:9], treasury[:,9]
X= X.astype(np.float)
y= y.astype(np.float)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'blue']
val = ["True","False"]

for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(val)
plt.title("Treasury Data Set")
plt.xlabel('Price Crossing')
plt.ylabel('Price Distortion')
plt.show()
n=np.arange(1,101)
train_accuracy = np.empty(len(n))
test_accuracy = np.empty(len(n))
from sklearn import neighbors 
for k in range(1,100):
    knn = neighbors.KNeighborsClassifier(k ,weights='uniform')
    knn.fit(X_train, y_train)
    train_accuracy[k] = knn.score(X_train,y_train)
    test_accuracy[k] = knn.score(X_test,y_test)

import matplotlib.pyplot as plt
plt.title("K-NN : Varying number of neighbors")
plt.plot(n,test_accuracy,label="Test Accuracy")
plt.plot(n,train_accuracy,label="Train Accuracy")
plt.legend()
plt.xlabel('No of Neighbors')
plt.ylabel('Acuracy')
plt.show()

print("The best test accuracy is ",max(test_accuracy),"at k = ",np.argmax(test_accuracy)+1)
plt.scatter(train_accuracy,test_accuracy, c='blue')
plt.title("Training vs Testing accuracies")
plt.xlabel('Training Accuracies')
plt.ylabel('Testing Accuracies')
plt.show()
print("\n\nThe best test accuracy is ",max(test_accuracy),"at k = ",np.argmax(test_accuracy)+1)

print("My name is Ricardo Lopez")
print("My NetID is: Juanrl2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

