from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, GlobalAveragePooling1D

# import pandas as pd

#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)



from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
# (23, 10) (419, 10)
x_train = x_train.reshape(419,10,1,1) # 이미지 4차원 데이터도 순서변경없이 차원수를 낮춰 DNN연산가능
x_test = x_test.reshape(23, 10, 1,1)

#2.모델 구성
model = Sequential()

model.add(Conv1D(128,2 ,padding = 'valid' ,input_shape=(10,1)))
model.add(Conv1D(64,2,padding = 'same' ,activation = 'relu'))
model.add(Conv1D(32,2,padding = 'same' ,activation = 'relu'))
model.add(MaxPooling1D())
model.add(Conv1D(16,2,padding = 'valid' ,activation = 'relu'))
model.add(Conv1D(16,2,padding = 'same' ,activation = 'relu'))
model.add(Conv1D(1,2,padding = 'same' ,activation = 'relu'))

model.add(Flatten())#(N, 180)
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print('loss : ', loss)
y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)
print("걸린시간",end)

'''
StandardScaler
loss :  4833.31103515625
r2score  0.3519392198190815
걸린시간 6.860862493515015

RobustScaler
loss :  4639.46923828125
r2score  0.37792988832977814
걸린시간 5.693086624145508

MinMaxScaler
loss :  3625.48193359375
r2score  0.5138875049881018
걸린시간 7.770100831985474
'''