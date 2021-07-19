import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';' , index_col=None, header=0) # ./ 현재폴더 ../ 상위폴더 데이터 구분자 ;
# index는 없고 헤더는 첫번째 라인


print(datasets.shape) # (4898, 12)

x = datasets.iloc[:,0:11]
y = datasets.iloc[:,[11]]


'''
변한것
output dimension
(4898, 1)
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
(4898, 7)
원핫 인코딩은 데이터의 분포를 벡터화 시킨것입니다.
원핫 인코딩을 통해서 output dimension이 바뀌습니다.
x = datasets.iloc[:,0:11]
y = datasets.iloc[:,[11]]
print(y.shape)
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
OE.fit(y)
y = OE.transform(y).toarray()
print(y)
print(y.shape)

배열 받는것
iloc 함수 행번호를 이용해서 행을 가져오는것 (마지막행 -1로도 가져옴)

OneHotEncoder
판다스를 넘파이로 바꾸고 xy분리후 y라벨확인 np.unique(y)
다중분류
모델링후 0.8이상

'''
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
OE.fit(y)
y = OE.transform(y).toarray()
'''
데이터의 분포가 012가 아닐때 케라스를 사용하면 012부터 채워줘 라벨수가 많아진다
그래서 모델의 분석이 정확하지 않을 수 있다
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.9995, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(x_test.shape, x_train.shape)
