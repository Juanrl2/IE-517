
import sklearn
import numpy as np
import pandas as pd
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

from sklearn.tree import DecisionTreeClassifier, plot_tree
dtc = DecisionTreeClassifier(max_depth=3)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
y_train_pred = dtc.predict(X_train)
from sklearn import metrics
accuracy= (metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
plot_tree(dtc, filled=True, fontsize =5)
plt.show()
plt.show()

print("The test accuracy is ",accuracy)
print( "Training accuracy ",metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.classification_report(y_test, y_pred) )
print( metrics.confusion_matrix(y_test, y_pred) )
print("My name is Ricardo Lopez")
print("My NetID is: Juanrl2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

