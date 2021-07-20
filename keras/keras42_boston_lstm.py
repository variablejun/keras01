
from re import M
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input
import numpy as np 

from sklearn.datasets import load_boston
datasets = load_boston()

#1.data
x = datasets.data
y = datasets.target
'''

'''
print(np.max(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

x_train = x_train.reshape(480,13,1) #
x_test = x_test.reshape(26, 13, 1)

'''

'''



#2.모델 구성 1D이용시 차원수부족 오류
model = Sequential()
model.add(LSTM(16,activation = 'relu',input_shape=(13,1)))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1 ,activation='relu'))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss)
y_pred = model.predict(x_test) 

# y_pred = scaler.transform(y_pred)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

'''
걸린시간 21.428115844726562
loss :  20.355998992919922
r2score  0.8421284246602054
'''