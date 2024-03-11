from sklearn.datasets import load_boston

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
boston=load_boston()
print(boston.feature_names)
print(boston.DESCR)

X=boston.data
y=boston.target

print(X.shape)
#print(y)
import pandas as pd
bos=pd.DataFrame(boston.data)
print(bos.head())

lm=LinearRegression()
lm.fit(X,y)

p=lm.predict(X)
#print(p)

print(np.sqrt(mean_squared_error(y,p)))
print(mean_absolute_error(y,p))
print(r2_score(y,p))

print(lm.coef_)
print(lm.intercept_)


#plt.scatter(y,p)
#plt.show()






