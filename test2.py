import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('GlobalTemperatures.csv',parse_dates=['date'])
DateColumn = df['date']
TemperatureColumn = df['LandAverageTemperature']


  
X =df.groupby(df.date.dt.year)['LandAverageTemperature'].mean()
Y = X.tolist()
X1 = X.index.tolist()


x = np.array(X1)
y = np.array(Y)
plt.scatter(x,y)
plt.show()
xreshape=x.reshape(-1,1)


model = LinearRegression().fit(xreshape, y)

r_sq = model.score(xreshape, y)
print(f"coefficient of determination: {r_sq}")



#Fazer a media mensal  já tem
#Para trabalhar com meses ao inves de variação diaria 