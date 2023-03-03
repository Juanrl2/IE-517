import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

df=pd.read_csv('C:/Users/juanrl2/Downloads/ME/MLF_GP1_CreditScore.csv')

rank=pd.read_csv('C:/Users/juanrl2/Downloads/ME/Look up.csv')

temp=[]
for i in range(len(df['Rating'])):
    for j in range(len(rank['Rating'])):
        if (np.char.lower(str(df['Rating'].values[i]))==rank['Rating'].values[j]):
             temp.append(float(rank['rank'].values[j]))

df['Rating']=temp
pd_corr=df.corr()
sns.heatmap(pd_corr)
plt.show()

corr_labels=['Rating','CFO','CFO/Debt','ROE','Free Cash Flow', 'Current Liabilities','Cash','Current Liquidity']
sns.heatmap(df[corr_labels].corr())
plt.show()
print("As we see from the above correleation plots, there is strong correlation between few of the features")


df2=df.values
X=df2[:,:-1]
y=df2[:,-1]
pca_test=PCA()
pca_test.fit(X)
plt.bar(range(pca_test.n_components_),pca_test.explained_variance_ratio_)
plt.plot(range(pca_test.n_components_),np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('component')
plt.ylabel('explained variance ratio')
plt.title('Variance ratio vs pca component')
plt.show()

score_svc=[]
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=5)), ('svc', SVC(gamma=10) )])
    pipe.fit(X_train,y_train)
    score_svc.append(pipe.score(X_test,y_test))
    y_predict=pipe.predict(X_test)
    
score_knn=[]
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=5)), ('knn', KNeighborsClassifier(n_neighbors=5) )])
    pipe.fit(X_train,y_train)
    score_knn.append(pipe.score(X_test,y_test))
    y_predict=pipe.predict(X_test)
    
score_dt=[]
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
    pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA(n_components=5)), ('dec_tree', tree.DecisionTreeClassifier(max_depth=4) )])
    pipe.fit(X_train,y_train)
    score_dt.append(pipe.score(X_test,y_test))
    y_predict=pipe.predict(X_test)
