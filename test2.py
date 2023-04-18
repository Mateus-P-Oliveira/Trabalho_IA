import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
df = pd.read_csv('GlobalTemperatures.csv')
DateColumn = df['dt']
TemperatureColumn = df['LandAverageTemperature']

#def to_integer(DateColumn):
    #return 10000*DateColumn.year + 100*DateColumn.month + DateColumn.day

tmp = DateColumn.str.split('-')
DateColumn = (tmp.str[0]+tmp.str[1]+tmp.str[2]).astype(int)

#print(DateColumn)
#print(TemperatureColumn)
reshape((-1, 1))
model = LinearRegression()
x= DateColumn
model = LinearRegression().fit(DateColumn, TemperatureColumn)
#Fazer a media mensal  já tem
#Para trabalhar com meses ao inves de variação diaria 