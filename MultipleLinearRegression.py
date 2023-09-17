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


# data = sns.load_dataset("studentf-mat.csv")
data = pd.read_csv("studentf-mat.csv" , sep=";")
print(data.head())
#    health  absences  G1  G2  G3
# 0       3         6   5   6   6
# 1       3         4   5   5   6
# 2       3        10   7   8  10
# 3       5         2  15  14  15
# 4       5         4   6  10  10

predict = "health" 
# x = np.array(data.drop([predict],axis = 1)) jupyter command line 
x = data.drop(columns=[predict], axis=1)
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

# visualization
sns.pairplot(x)
plt.show()

















