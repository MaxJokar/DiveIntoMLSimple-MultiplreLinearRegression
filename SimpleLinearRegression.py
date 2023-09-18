"""
 Solve a usecase using multiple linear regression
"""

import numpy as np
import pandas as pd
import sklearn
# from sklearn import linear_model
import matplotlib.pyplot as plt
# # import matplotlib as mpl
import seaborn as sns
import matplotlib_inline
from sklearn.model_selection import train_test_split

# data = sns.load_dataset("studentf-mat.csv")
data = pd.read_csv("height-weight.csv" , sep=";")
# print(data.head())
#    weight  height
# 0     130      26
# 1     135      28
# 2     139      32
# 3     150      35



#  scatter is a function which helps you to project your data points respectively with
# x and y axis
# scatter plot 
# plt.scatter(data["weight"], data["height"])
# plt.xlabel("weight")
# plt.ylabel("height")
# plt.show()

#  Based on diagram to know whether this is positive or negative relationship 
# print(data.corr())
#           weight    height
# weight  1.000000  0.970105  ==> means highly positive correlated 
# height  0.970105  1.000000

 

# seaborn for visualization :
# sns.pairplot(data)
# plt.show()

# independant & dependant featurs 
#  for independant feature we name is as  x 
#  independant feature should be data frame feature must be form of dataframe or 2 dimension array 
X = data[["weight"]]   # independant feature  : 2 dimenstion 
# print(X) 
# 0    130
# 1    135
# 2    139
# 3    150
# 4    154
# 5    159
# 6    178
# Name: weight, dtype: int64

rows =np.array(X).shape
# print("this is rows : ",rows)
# this is rows :  (7, 1)
# (7, 1)  row, column , in Multiregression column  will increase
#  wihtou double [[]]when we shape we wont get any info about column 
y = data["height"]   # dependant feature , this variable can be in series or  1d array

#  train for training  data and test for  checking how the model is predicting the new data 
#  train test split : for predict and fit 
# model uses train for trainig that data 
# for testing or seeing how the model preforms for the new data we use test
# random...take data and put in trainig data ranodmizely
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y ,
                                            test_size=0.25, random_state = 42)

# print("this is x_train :",X_train.shape)
# this is x_train : (5, 1)  i have 1 feature  , 5 rows

#  Standardization :our input feature because our gradient descent gets applied into the independant feature
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# print(scaler.fit_transform(X_train))
#  between 0 and  1 ==> all datapoint transformed in 0 ,1 scale 
# [[ 0.23397548]
#  [-1.32586106]
#  [-0.15598365]
#  [-0.46795096]
#  [ 1.71582019]]

#  for X_data we dont apply fit_transform difference for fit transform and 
X_train = scaler.fit_transform(X_train)
# print("xtrain for  standardization ",X_train)
X_test = scaler.transform(X_test)
# print("x_test for  standardization ",X_train)

# xtrain for  standardization  [[ 0.23397548]
#  [-1.32586106]
#  [-0.15598365]
#  [-0.46795096]
#  [ 1.71582019]]
# x_test for  standardization  [[ 0.23397548]
#  [-1.32586106]
#  [-0.15598365]
#  [-0.46795096]
#  [ 1.71582019]]

# Apply simple Linear regression 
from sklearn.linear_model import LinearRegression
# first we must initialize the object 
regression = LinearRegression()
#  whatever train data we give should be 2d in output is ok but here NO
regression.fit(X_train, y_train)
# how many processors are when I execute 
LinearRegression(n_jobs= -1)
coef = regression.coef_
# print("my coefficient or slope  :" , coef)
# my coef : [12.69706942]  ==> y=B0 + B1x1  B1 is our coef
#  to predict testdata 

#  intercept 
intercep = regression.intercept_
# print("intercept is   :" , intercep)
# intercept is   : 44.8
#  SUMMARY from above "importance of coef and slope "
# one unit in weight value leads to 12.69 unit movement in height value 


#  Plot Training data plot best fit line
plt.scatter(X_train , y_train)
# gives my predixted value along with the straight line 
plt.plot(X_train, regression.predict(X_train))
# plt.show()


#  Prediction for test data :x_test_prediction
y_pred = regression.predict(X_test)
# print("prediction x_test :" , y_pred)
# prediction x_test : [19.05304136 24.00437956]



#  Prediction of test data 
#  predicted height output  = intecept 44.8 + coef_ (weight) 12.69
# y_pred_test =
y_pred =  regression.predict(X_test)



# Performance Metrics
from sklearn.metrics import mean_absolute_error , mean_squared_error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse=np.sqrt(mse)
# print("this is  mse :", mse)
# print("this is  mae :", mae)
# print("this is  rmse :", rmse) 
# this is  mse : 32.11260849746337
# this is  mae : 5.4712895377129005
# this is  rmse : 5.666798787451639

# Assumption :
# R^2 =1-SSR/SST
# coefficient of determiantion SSR = sum of squares of residuals SST = total sum of swares
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("this is score ",score)
# this is score  -31.11260849746337


# display and adjusted R-squared 
print(1- (1-score)*(len(y_test)-1)/ (len(y_test)-X_test.shape[1]-1))

# To Be Continued,...