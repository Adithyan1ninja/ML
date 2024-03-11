

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("Advertising.csv")

print(data.info())
print(data.describe())
#print(data.head())
data = data.to_numpy()
X = data[:,1:4]
y = data[:,-1] 




X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

model = LinearRegression()
model.fit(X_train,y_train)

p = model.predict(X_test)

plt.scatter(y_test,p)
plt.show()

