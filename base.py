import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('GlobalTemperatures.csv')
DateColumn = df['date']
TemperatureColumn = df['LandAverageTemperature']
tmp = DateColumn.str.split('-')
DateColumn = (tmp.str[2]+tmp.str[1]+tmp.str[0]).astype(int)

TemperatureColumn = TemperatureColumn.fillna(0)

x= DateColumn.to_numpy().reshape((-1,1))
y= TemperatureColumn.to_numpy()

#x.reshape((1,-1))
print(x)
print(y)
plt.scatter(x, y)
plt.xlabel("Data")
plt.ylabel("Temperatura Celsius")
plt.show()

model = LinearRegression().fit(x, y)
#sns.histplot(model)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")