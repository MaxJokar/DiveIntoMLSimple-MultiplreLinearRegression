"""
 Solve a usecase using multiple linear regression
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
# # import matplotlib as mpl
import seaborn as sns
import matplotlib_inline



# data = sns.load_dataset("studentf-mat.csv")
data = pd.read_csv("studentf-mat.csv" , sep=";")
print(data.head())
#    health  absences  G1  G2  G3
# 0       3         6   5   6   6
# 1       3         4   5   5   6
# 2       3        10   7   8  10
# 3       5         2  15  14  15
# 4       5         4   6  10  10

health = "health" 
# x = np.array(data.drop([health],axis = 1)) jupyter command line 
x = data.drop(columns=[health], axis=1)
print(x)
#     absences  G1  G2  G3
# 0          6   5   6   6
# 1          4   5   5   6
# 2         10   7   8  10
# 3          2  15  14  15
# 4          4   6  10  10
# 5         10  15  15  15
# 6          0  12  12  11

# sns.jointplot(af = data,  y  = "G1", hue = "G2" , hue_norm="G3" ) Jupyter 

# visualization the dotpoints more closely  
# sns.pairplot(x)
# plt.scatter(x["G3"],x["absences"], color = 'r')
# plt.xlabel("G3")
# plt.ylabel("absences")
# plt.show()


# independant & dependant featurs 
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]

# print("independant & dependant featurs  \n")
# print(X.head(),y)
#    health  absences  G1  G2
# 0       3         6   5   6
# 1       3         4   5   5
# 2       3        10   7   8
# 3       5         2  15  14
# 4       5         4   6  10
# 25     8
# 26    11
# 27    15
# Name: G3, dtype: int64


# train & test split 
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# sns.regplot(x = "G3" , y = "absences" , data = data)
# plt.show()



scaler = sklearn
X_train = scaler.fit_transform(X_train)

print(X_train)
