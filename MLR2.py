
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('datafile.csv')
X=dataset.iloc[:,-4:-1:1].values
Y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

print(y_test,y_pred)
#print(np.concatenate(y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1)
