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
깃허브 참고해서 채우자
안돌아감 ㅜㅠ
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
train_size = 0.95, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(50,activation='relu', input_dim = 11))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # 이진분류모델 에 대한 로스
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=5, mode='max', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.3, callbacks=[es]) 

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[2])


print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)


