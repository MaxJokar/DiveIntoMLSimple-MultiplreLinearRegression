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
data = pd.read_csv("studentf-mat.csv" , sep=";")
# print(data.head())
#    health  absences  G1  G2  G3
# 0       3         6   5   6   6
# 1       3         4   5   5   6
# 2       3        10   7   8  10
# 3       5         2  15  14  15
# 4       5         4   6  10  10

# health = predict or y 
health = "health" 
# x = np.array(data.drop([health],axis = 1)) jupyter command line 
x = data.drop(columns=[health], axis=1)
# print("this is  x \n", x)
# Or 
# independant & dependant featurs 
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]
#     absences  G1  G2  G3
# 0         6   5   6   6
# 1         4   5   5   6
# 2        10   7   8  10
# 3         2  15  14  15
# 4         4   6  10  10
# 5        10  15  15  15
# 6         0  12  12  11

# health = predict or y 
health_predict_y = np.array(data[health])
# print("this is health_predict_y :",health_predict_y)
# this is y : [3 3 3 5 5 5 3]
# print("="*50)

#  train test split : for predict and fit 
# model uses train for trainig that data 
# for testing or seeing how the model preforms for the new data we use test
# random...take data and put in trainig data ranodmizely
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,health_predict_y ,
                            test_size=0.1, random_state = 42)
print("this is xtrain .shape :",x_train.shape)
# this is xtrain .shape : (6, 4) I have 4 feature with 6 rows 



# sns.jointplot(af = data,  y  = "G1", hue = "G2" , hue_norm="G3" ) Jupyter 

# visualization
# sns.pairplot(x)
# plt.show()

#  can do only with 2 features for  x and y  axis
# sns.regplot(x = "G3" , y = "absences" , data = data)
# plt.show()



# the industry's go-to algorithm. 🙂 StandardScaler 
# standardizes a feature by subtracting the mean and then scaling to unit variance

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
# print(x_train)

# ==================================================
# [[ 0.1767767  -1.11456054 -1.03279556 -1.22098221]
#  [-1.41421356  0.49963059  0.51639778  0.13566469]
#  [ 1.23743687  1.19142679  1.29099445  1.22098221]
#  [-0.88388348  1.19142679  1.03279556  1.22098221]
#  [-0.35355339 -1.11456054 -1.29099445 -1.22098221]
#  [ 1.23743687 -0.65336308 -0.51639778 -0.13566469]]



from sklearn.linear_model import LinearRegression
#  our model is regression 
regression =LinearRegression()
regression.fit(x_train, y_train)
# print(LinearRegression)
# <class 'sklearn.linear_model._base.LinearRegression'>


# Cross validation with the model that we have created 
# EVALUATE SCORE :Cross_val_score is a method which runs cross validation on a dataset 
# to test whether the model can generalise over the whole dataset.
#  with the help of cross validation I get 5 differenct accuracy =MSE1 , 2....5 
from sklearn.model_selection import cross_val_score
validation_score =  cross_val_score(regression, x_train ,y_train  ,scoring='neg_mean_squared_error',
                                    cv = 3 )

# print("validation score is : ",validation_score)
# validation score is :  [-2.69824032 -3.77659459 -3.16156184]

# print("validation score  using cross validation is : \n",np.mean(validation_score))
#   mean validation score by performing  cross validation is : -3.212132247747361


# Prediction :
y_pred = regression.predict(x_test)
# print("Prediction for  y_pred is :",y_pred)
# Prediction for  y_pred is : [3.66666667]



# Performance metrics 
from sklearn.metrics import mean_absolute_error, mean_squared_error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse=np.sqrt(mse)
# print("this is  mse :", mse)
# print("this is  mae :", mae)
# print("this is  rmse :", rmse)
# this is  mean_squared_error : 1.7777777777777795
# this is  mean_absolute_error : 1.333333333333334
# this is  rmse : 1.333333333333334



# Assumption :
# R^2 =1   SSR/SST
# coefficient of determiantion SSR = sum of squares of residuals SST = total sum of swares
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
# print("this is score ",score)
# display and adjusted R-squared 
# print(1- (1-score)*(len(y_test)-1)/ (len(y_test)-x.shape[1]-1))









