import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('GlobalTemperatures.csv',parse_dates=['date'])
DateColumn = df['date']
TemperatureColumn = df['LandAverageTemperature']


  
X =df.groupby(df.date.dt.year)['LandAverageTemperature'].mean()
Y = X.tolist()
X1 = X.index.tolist()


x = np.array(X1)
y = np.array(Y)
#plt.scatter(x,y)
#plt.show()
xreshape=x.reshape(-1,1)

x_train = xreshape[50:]#Slice do Array
x_test = xreshape[30:]#Slice do Array

y_train = y[50:]#Slice do Array
y_test = y[30:]#Slice do Array
#plt.scatter(x_test,y_test)
#plt.scatter(x_train,y_train)
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

plt.scatter(x_train, y_train, color="blue")
#plt.scatter(x_test, y_test, color="orange")
#plt.scatter(x,y)
plt.plot(x_test, y_pred, color="blue", linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.show()

#model = LinearRegression().fit(xreshape, y)

#r_sq = model.score(xreshape, y)
#print(f"coefficient of determination: {r_sq}")




