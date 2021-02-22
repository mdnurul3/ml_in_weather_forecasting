# Load libraries
import numpy as np
import matplotlib.pyplot as plt
#from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
# Load dataset

import timeit
start = timeit.default_timer()

# Split-out validation dataset
df = pd.read_excel('pollution_weather_raw_2020_2_col.xlsx')
df.columns = ['Date', 'Temp in C']
#print (df)

df["Date"] = df["Date"].dt.strftime('%Y%m%d').astype(float)

X=df["Date"];
X = X.values
X = np.asarray(X[:])
y=df["Temp in C"];
y = y.values
y = np.asarray(y[:])

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.30, random_state=1)
X = X.reshape(-1,1)
y = y.reshape(-1,1)
X_train = X_train.reshape(-1,1)

#Predicting using SVR
svr_poly = SVR(kernel='rbf', C=100, gamma='auto', degree=2, epsilon=.1,
               coef0=1)

svr_poly.fit(X_train,Y_train)
save_svr=svr_poly.predict(X)
#print(save_svr)
#Plotting

plt.plot(df["Date"],y,'b-',df["Date"],save_svr,'g-')
plt.title('Year vs Tempereture')
plt.xlabel('Year')
plt.ylabel('Temperature(k)')
plt.legend(['Actual','Prediction'])
plt.xticks(rotation=45)
#plt.legend('Actual',"ML(SVR) prediction")
plt.savefig('Year vs Tempereture')
#plt.show()
rmse = sqrt(mean_squared_error(y,save_svr))
print('Test RMSE: %.3f' % rmse)
stop = timeit.default_timer()

print('Run Time: ', stop - start,"Seconds")