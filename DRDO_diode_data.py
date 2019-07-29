# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 01:24:37 2019

@author: Prashun
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Diode Data.csv')
X = dataset.iloc[:, [3, 4]].values
y = dataset.iloc[:, [2]].values
          
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising Actual Set
plt.scatter(dataset['V'], np.log(dataset['RD']), color='red')
plt.title('Voltage Vs Dynamic Resistance(Actual Set)')
plt.xlabel('Voltage')
plt.ylabel('Dynamic Resistance(logarithmic Scale)')
plt.grid(True)
plt.show()

plt.scatter(dataset['I'], np.log(dataset['RD']), color='red')
plt.title('Current Vs Dynamic Resistance(Actual Set)')
plt.xlabel('Current')
plt.ylabel('Dynamic Resistance(logarithmic Scale)')
plt.grid(True)
plt.show()

# Visualising Test Set
plt.scatter(X_test[:,1],np.log(y_test), color='red')
plt.title('Current Vs Dynamic Resistance(Test Set)')
plt.xlabel('Current')
plt.ylabel('Dynamic Resistance(logarithmic Scale)')
plt.grid(True)
plt.show()

plt.scatter(X_test[:,0], np.log(y_test), color='red')
plt.title('Voltage Vs Dynamic Resistance(Test Set)')
plt.xlabel('Voltage')
plt.ylabel('Dynamic Resistance(logarithmic Scale)')
plt.grid(True)
plt.show()

#Showing the difference in Prediction Vs Test Results
plt.scatter(y_test, np.log(y_pred), color='green')
plt.title('Prediction Vs Test Results')
plt.xlabel('Test Results')
plt.ylabel('Prediction Results(logarithmic Scale)')
plt.grid(True)
plt.show()



