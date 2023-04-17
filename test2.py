import pandas as pd
df = pd.read_csv('GlobalTemperatures.csv')
DateColumn = df['dt']
TemperatureColumn = df['LandAverageTemperature']

print(DateColumn)
print(TemperatureColumn)

#Fazer a media mensal 
#Para trabalhar com meses ao inves de variação diaria 