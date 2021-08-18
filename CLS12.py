import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('datafile1.csv')
X=dataset.iloc[:,-1].values
Y=dataset.iloc[:,0].values
Y1=Y
print(type(X))

Y=pd.get_dummies(Y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
Y=np.array(ct.fit_transform(Y))
#print(Y1,Y)
import array
X=X.reshape(-1,1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier.fit(x_train,y_train)

clf = classifier.predict([[9.83]])
print(clf)

encodings = {
[[1 0 0 0 0 0 0 0 0 0 0]]:'Arhar',
[[0 1 0 0 0 0 0 0 0 0 0]]: 'Cotton',
[[0 0 1 0 0 0 0 0 0 0 0]]:'Gram',
[[0 0 0 1 0 0 0 0 0 0 0]]: 'Groundnut',
[[0 0 0 0 1 0 0 0 0 0 0]]: 'Maize',
[[0 0 0 0 0 1 0 0 0 0 0]]: 'Moong',
[[0 0 0 0 0 0 1 0 0 0 0]]: 'Paddy',
[[0 0 0 0 0 0 0 1 0 0 0]]: 'Rapeseed',
[[0 0 0 0 0 0 0 0 1 0 0]]: 'SugarCane',
[[0 0 0 0 0 0 0 0 0 0 1]]: 'Wheat'}





