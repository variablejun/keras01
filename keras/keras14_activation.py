from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import time
# import pandas as pd

#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.4, random_state=66)

print(np.shape(x), np.shape(y))
print(datasets.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
start = time.time()

#2.모델 구성
models = Sequential()
models.add(Dense(501,activation='relu',input_shape=(10,)))
models.add(Dense(401, activation='relu'))
models.add(Dense(301, activation='relu')) # 활성함수
models.add(Dense(201, activation='relu'))
models.add(Dense(101, activation='relu'))
models.add(Dense(51, activation='relu'))
models.add(Dense(25, activation='relu'))
models.add(Dense(13, activation='relu'))
models.add(Dense(7, activation='relu'))
models.add(Dense(3))

models.add(Dense(1))

'''

r2score  0.518597407069652
r2score  0.5176675166315408
r2score  0.4706347474154573
'''

#3.complie/훈련
models.compile(loss = 'mse', optimizer = 'adam')
models.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.3) 

#4.평가/예측
loss = models.evaluate(x_test, y_test) 
print('loss : ', loss)


y_pred = models.predict(x_test) 


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)
end = time.time() - start
print('걸린 시간 : ', end)
'''
0.62 이상올리기 과제
r2score  0.5037277375596032
'''