import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from math import sqrt

df = pd.read_csv('GlobalTemperatures.csv',parse_dates=['date'])
DateColumn = df['date']
df = df[df['LandAverageTemperature'].notna()] #Retina os NaN da temperatura
df_columns = df.columns #Pega o nome das colunas
TemperatureColumn = df['LandAverageTemperature']
columns = df.columns #Faz uma lista com as colunas

array = [2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,2044,2045,2046,2047,2048,2049,2050,2051,2052,2053]
dateList =np.array(array).reshape(-1,1) #Isso modifica o array acima
#sns.set(rc={'figure.figsize':(20,8)}, style='whitegrid') #Checa se existe outliners
#sns.boxplot(data=df) 
fig, axes = plt.subplots(3,3)
plt.subplots_adjust(hspace = 0.4)
for i, column in enumerate(columns[1:9]):
    stats.probplot(df[column], dist="norm", plot=axes[int(i/3)][i%3])
    axes[int(i/3)][i%3].set_title(column)
plt.show()

#Plot de histograma
fig, axes = plt.subplots(3,3)
for i, column in enumerate(columns[1:9]):
    temp_df = pd.DataFrame(df[column], columns=[column])
    temp_df.hist(ax=axes[int(i/3)][i%3])
plt.show()
#-------------------------Validação por mês
X =df.groupby(df.date.dt.month)['LandAverageTemperature'].mean()
Y = X.tolist()
X1 = X.index.tolist()

#------------------------------
plt.xlabel("Ano")
plt.ylabel("Temperatura Celsius")
x = np.array(X1)
y = np.array(Y)
plt.scatter(x,y,color="orange")
plt.show()
#-----------------------
X =df.groupby(df.date.dt.year)['LandAverageTemperature'].mean()
Y = X.tolist()
X1 = X.index.tolist()

x = np.array(X1)
y = np.array(Y)

xreshape=x.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(xreshape, y, test_size=0.2, random_state=85) #Novo metodo de treino

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
print("Coefficients: \n", regr.coef_)

# Mean Squred Error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error(Testing) :', round(mse, 4))

# Root Mean Squared Error
rmsq = sqrt(mean_squared_error(y_test, y_pred))
print('Root mean squared error(Testing) :', round(rmsq, 4))


plt.scatter(x,y,color="orange")
plt.plot(x_test, y_pred, color="blue", linewidth=1)
plt.xlabel("Ano")
plt.ylabel("Temperatura Celsius")
future_predictions = regr.predict(dateList)
print("Future Predictions: \n",future_predictions)
plt.plot(dateList,future_predictions,color="red", linewidth=1)
plt.show()

sns.set(rc={'figure.figsize':(20,10)}) # O quanto a temperatura esperada varia da temperatura media
plt.plot(y_pred[:200], label='Predicted', color='blue')
plt.plot(y_test[:200], label='Actual', color='red')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('Predicitons vs Actual')
plt.legend()
plt.show()

