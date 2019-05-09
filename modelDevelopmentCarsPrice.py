# -*- coding: utf-8 -*-
"""
Creating models
Dataset used is of various car attributes and the overall price of that car.
Models a car can have a predicted price given some attributes if the right model is used.

Final conclusion is at the bottom.

@author: Sphyncx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path of data
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head() #check



###############################################################################
# Linear Regression
###############################################################################
from sklearn.linear_model import LinearRegression

# Simple linear regression
# Yhat = a + bx
lm = LinearRegression()  #create linear regression object
lm

# Trying to predict car price based on highway mpg of cars
X = df[['highway-mpg']]
Y = df[['price']]

# Fit the linear model
lm.fit(X,Y)

# Output the prediction
Yhat = lm.predict(X)
Yhat[0:5]

# intercept
lm.intercept_

# slope
lm.coef_
# price = 38423.31 - 821.73 x highway-mpg

###############################################

# Multiple Linear Regression
# Yhat = a + b1x1 + b2x2 + b3x3 + ...
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']] # predictor vars

# Fit the linear model
lm.fit(Z, df['price'])

# intercept
lm.intercept_

# b1, b2, b3, b4 coefficients
lm.coef_
# price = -15806.62 + 53.5 x horsepower + 4.71 x curb-weight + 81.53 x engine-size + 36.06 x highway-mpg



###############################################################################
# Regression Plot
###############################################################################
import seaborn as sns
%matplotlib inline 

width = 12
height = 10

plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

df[["peak-rpm","highway-mpg","price"]].corr()
# The variable "peak-rpm" has a stronger correlation with "price", it is 
# approximate -0.704692 compared to "highway-mpg" which is approximate -0.101616.



###############################################################################
# Residual Plot
###############################################################################
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
# a non-linear model would work best

# prediction
Y_hat = lm.predict(Z)

plt.figure(figsize=(width, height))

ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()



###############################################################################
# Polynomial Regression
###############################################################################
# Y = a + b1x^2 + b2x^2 + b3x^3 + ...
# Quadratic (2nd order [b2]), Cubic (3rd order [b3]), etc
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'highway-mpg')

np.polyfit(x, y, 3)


from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2) # object of degree 2
pr
Z_pr = pr.fit_transform(Z)

Z.shape # original data:  201 samples and 4 features
Z_pr.shape # after transformation:  201 samples and 15 features



###############################################################################
# Pipelines
###############################################################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline
# creating a list a tuples that include the name of the model and its constructor
Input = [('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# input into pipeline constructor
pipe = Pipeline(Input)
pipe

# normalize data, transform, and fit the model
pipe.fit(Z,y)

# normalize data, transform, and produce a prediction
ypipe = pipe.predict(Z)
ypipe[0:4]



###############################################################################
# R^2 (higher means better fit) & Mean Squared Error (lower means better fit)
###############################################################################
# Simple linear regression
lm.fit(X,Y) # highway-mpg, price
print('R-square is: ', lm.score(X,Y)) # R^2
# ~49.659% of the variation of the price is explained by this model

# Mean Squared Error
Yhat = lm.predict(X)
print('The output of the first 4 predicted values are: ', Yhat[0:4])

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
# R^2: 0.49659
# MSE: 31635042.9446

###############################################

# Multiple linear regression
lm.fit(Z, df['price']) # horsepower, curb-weight, engine-size, highway-mpg
print('R-square is: ', lm.score(Z, df['price'])) # R^2
# ~80.896%

# Mean Squared Error
Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', mean_squared_error(df['price'], Y_predict_multifit))
# R^2: 0.80896
# MSE: 11980366.8707

###############################################

# Polynomial Fit
from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))
print('R-square is: ', r_squared)
# ~67.419%

# Mean Squared Error
mean_squared_error(df['price'], p(x))
# R^2: 0.67419
# MSE: 20474146.4263



###############################################################################
# Prediction and Decision Making
###############################################################################
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline 

# create new input
new_input=np.arange(1, 100, 1).reshape(-1, 1)

lm.fit(X, Y) # highway-mpg, price
lm

# prediction
yhat=lm.predict(new_input)
yhat[0:5]

# plot
plt.plot(new_input, yhat)
plt.show()



###############################################################################
# Conclusion:
# Multiple Linear Regression is the best model for this data in order to predict
# price from the dataset.
# Simple Linear regression < polynomial fit < multiple linear regression
# in this case because of the MSE was smaller and R-squared is larger.
###############################################################################
