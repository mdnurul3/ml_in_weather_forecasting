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
df = pd.read_excel('pollution_weather_raw_2020_2_col.xls')
df.columns = ['Date', 'Temp in C']




print(df.describe())
