import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
df = pd.read_csv('GlobalTemperatures.csv')
DateColumn = df['dt']
TemperatureColumn = df['LandAverageTemperature']

#def to_integer(DateColumn):
    #return 10000*DateColumn.year + 100*DateColumn.month + DateColumn.day

tmp = DateColumn.str.split('-')
DateColumn = (tmp.str[0]+tmp.str[1]+tmp.str[2]).astype(int)
TemperatureColumn = TemperatureColumn.fillna(0)
#print(DateColumn)
#print(TemperatureColumn)
#reshape((-1, 1))
#model = LinearRegression()
x= DateColumn.to_numpy().reshape((-1,1))
y= TemperatureColumn.to_numpy()

#x.reshape((1,-1))
print(x)
print(y)

model = LinearRegression().fit(x, y)
#sns.histplot(model)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
#Fazer a media mensal  já tem
#Para trabalhar com meses ao inves de variação diaria 