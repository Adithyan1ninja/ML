from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



data=pd.read_csv("Immunotherapy.csv")
print(data.head())

data = data.to_numpy()
X=data[:,:8]
y=data[:,-1]

print(X)
print(y)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
p=knn.predict(X)
print(p)

print(accuracy_score(y,p))
print(confusion_matrix(y,p))