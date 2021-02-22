import pandas as pd
import numpy as np
df = pd.read_excel('svm_test_data_2847.xls')
df.columns = ['Date', 'Temp in C']
#print (df)

df["Date"] = df["Date"].dt.strftime('%Y%m%d').astype(float)
print (df)
dateinfloat=df["Date"];
array = dateinfloat.values
X = np.asarray(array[0:10])
y = np.asarray(array[11:100])