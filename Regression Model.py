import pandas as pd
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


from sklearn.linear_model import LinearRegression
from statistics import mean
from sklearn.metrics import mean_squared_error

Boston = pd.read_csv('E:\DATASETS/BostonHousing.csv')
print(Boston)
print(Boston.corr())

# BUILDING A LINEAR REGRESSION MODEL
#Splitting dataset to X and Y variables
y=Boston['medv']
x=Boston.drop(['medv'], axis=1) #axis=1 is for side by side matrix

#Splitting the dataset into 80/20
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)
#Data Dimension
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# The Linear Reg Model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
model=linear_model.LinearRegression() #defining the regression model
model.fit(x_train, y_train) #Building a training Model
y_pred=model.predict(x_test)
print(y_pred) #Using the traind model to make a prediction
#Checking model performance
print('Coefficents:', model.coef_)
print('Intercept: ', model.intercept_)
print('MSE:  ', '{:.2f}'.format(mean_squared_error(y_test, y_pred)))
print('Coefficient of determination (R^2): ', '{:.2f}'.format(r2_score(y_test, y_pred)))
print(Boston.head())
#The Equation of the model
# medv=-4.59*(crim) + 5.20*(zn) + 3.08*(indus) + 3.53*(chas) + ... + 31.73

# #String formatting
# print(r2_score(y_test, y_pred))
#
# #Making a scatter plot
# import seaborn as sns
# print(y_test)
# print(y_pred)
#
# sns.scatterplot(y_test, y_pred, marker='*', alpha=0.3)
# plt.show()