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
from sklearn.preprocessing import LabelEncoder
# Load dataset



# Split-out validation dataset
dataset = pd.read_csv("svm_test_data_2847_convert.csv")
values = dataset.values
 
# integer encode direction
encoder = LabelEncoder()
values[:,0] = encoder.fit_transform(values[:,0])
# ensure all data is float
values = values.astype('float32')

X = np.asarray(values[:,0])
y = np.asarray(values[:,1])
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

plt.plot(y,X,'b-',save_svr,X,'g-')
plt.title('Height vs Pressure')
plt.xlabel('Pressure(Mb)')
plt.ylabel('Height(Km)')
plt.legend('Actual',"ML(SVR) prediction")
plt.savefig('Height vs Pressure Six Point')
#plt.show()


#This Section for different data to check prediction

#Importing Height Data
# dataset = pd.read_csv("hgt_vs_pres.csv")
# array = dataset.values
# X2 = np.asarray(array[29:57,0])
# X3 = np.asarray(array[29:57,1])
# X_train, X_validation, Y_train, Y_validation = train_test_split(X2, X3, test_size=0.30, random_state=1)
# X2 = X2.reshape(-1,1)
# X3 = X3.reshape(-1,1)
# X_train = X_train.reshape(-1,1)

# #Predicting using SVR
# svr_poly2 = SVR(kernel='rbf', C=100, gamma='auto', degree=2, epsilon=.1,
#                 coef0=1)
# svr_poly2.fit(X_train,Y_train)
# save_svr2=svr_poly2.predict(X2)
# print(svr_poly2)

# # with open('predicted_temp.txt', 'w') as output:
# #     output.write(save_svr2())
# #      output.close()
# #Plotting

# plt.plot(X3,X2/1000,'b-',save_svr2,X2/1000,'g-')
# plt.title('Height vs Pressure')
# plt.xlabel('Pressure(Mb)')
# plt.ylabel('Height(Km)')
# plt.legend('Actual',"ML(SVR) prediction")
# plt.savefig('Height vs Pressure partial')
#plt.show()

# scores = model.evaluate(X_validation, Y_validation, verbose=1)
# print("Accuracy: %.2f%%" % (scores[1]*100))



