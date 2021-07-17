import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';' , index_col=None, header=0) # ./ 현재폴더 ../ 상위폴더 데이터 구분자 ;
# index는 없고 헤더는 첫번째 라인


print(datasets.shape) # (4898, 12)

x = datasets.iloc[:,0:11]
y = datasets.iloc[:,[11]]
print(y.shape)
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
OE.fit(y)
y = OE.transform(y).toarray()
print(y)
print(y.shape)
'''
 fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  ...  total 
sulfur dioxide  density    pH  sulphates  alcohol
0               7.0              0.27         0.36            20.7      0.045  ...                 170.0  1.00100  3.00       0.45      8.8
1               6.3              0.30         0.34             1.6      0.049  ...                 132.0  0.99400  3.30       0.49      9.5
3               7.2              0.23         0.32             8.5      0.058  ...                 186.0  0.99560  3.19       0.40      9.9
4               7.2              0.23         0.32             8.5      0.058  ...                 186.0  0.99560  3.19       0.40      9.9
...             ...               ...          ...             ...        ...  ...                   ...      ...   ...        ...      ...
4893            6.2              0.21         0.29             1.6      0.039  ...                  92.0  0.99114  3.27       0.50     11.2
4894            6.6              0.32         0.36             8.0      0.047  ...                 168.0  0.99490  3.15       0.46      9.6
4895            6.5              0.24         0.19             1.2      0.041  ...                 111.0  0.99254  2.99       0.46      9.4
4896            5.5              0.29         0.30             1.1      0.022  ...                 110.0  0.98869  3.34       0.38     12.8
4897            6.0              0.21         0.38             0.8      0.020  ...                  98.0  0.98941  3.26       0.32     11.8

'''