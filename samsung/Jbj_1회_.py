import pandas as  pd
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Flatten, Conv1D
import decimal as dc
import tensorflow as tf

#1.data 
dataframe1 = pd.read_csv('../_data/SK주가 20210721.csv',encoding='cp949')
dataframe2 = pd.read_csv('../_data/삼성전자 주가 20210721.csv',encoding='cp949')
datasets1 =dataframe1.values
datasets2 =dataframe2.values

datasets1 = np.array(datasets1)
datasets2 = np.array(datasets2)

datasets1 = datasets1[:,[1,2,3,4,10]].astype(np.float32)
datasets2 = datasets2[:,[1,2,3,4,10]].astype(np.float32)

datasets1 = datasets1[0:2601,:]
datasets2 = datasets2[0:2601,:]
datasets1 = np.flip(datasets1,axis=0)
datasets2 = np.flip(datasets2,axis=0)

print("=======MODEL========")
print(datasets1)
print("=========sam========")
print(datasets2)
print("=======SPLIT========")
size = 6
def split_x(dataset, size):
     aaa=[]
     for i in range(len(dataset) - size + 1):
          subset = dataset[i : (i + size)].astype(np.int32)
          aaa.append(subset)

     return np.array(aaa)

datasets2 = split_x(datasets2, size)
datasets1 = split_x(datasets1, size)
print(datasets1)
print(datasets2)

x1 = datasets1[:, :5]
y1 = datasets1[:,5]
x2 = datasets2[:, :5]
y2 = datasets2[:,5]
print("=======x1======")
print(x1)
print("=======y1======")
print(y1)
print("=======x2======")
print(x2)
print("=======y2======")
print(y2)

print(x1.shape)
print(y1.shape)
print(x2.shape)
print(y2.shape)  


from sklearn.model_selection import train_test_split
x1_train, x1_test,x2_train,x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2,train_size = 0.7, random_state=66) # train_size 0.7
#(2596, 5, 5)
#(2596, 5)
#(2596, 5, 5)
#(2596, 5)
#(1817, 5, 5) (779, 5, 5) (1817, 5, 5) (779, 5, 5) (1817, 5) (779, 5) (1817, 5) (779, 5)
'''
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()

y1_test = y1_test.reshape(-1, 1)
y1_train = y1_train.reshape(-1, 1)
y2_test = y2_test.reshape(-1, 1)
y2_train = y2_train.reshape(-1, 1)
OE.fit(y1_test)
y1_test = OE.transform(y1_test).toarray() # 리스트를 배열로 바꾸어주는 함수
OE.fit(y1_train)
y1_train = OE.transform(y1_train).toarray()
OE.fit(y2_test)
y2_test = OE.transform(y2_test).toarray() # 리스트를 배열로 바꾸어주는 함수
OE.fit(y2_train)
y2_train = OE.transform(y2_train).toarray()
'''



#2.모델 구성


#3.complie/훈련


import time
starttime = time.time()

#model = load_model('/_save/Modelsave_samsung.h5')
model = load_model('/_save/MCP_samsung.hdf5')

loss = model.evaluate([x1_test,x2_test], [y1_test, y2_test]) 
end = time.time()- starttime

#4.평가/예측
#y_pred = model.predict(x2_test) 

#rint(y_pred)
print("걸린시간", end)
print('loss : ', loss)




'''

25/25 [==============================] - 0s 6ms/step - loss: 5645876068352.0000
걸린시간 120.96780347824097
loss :  5645876068352.0

Model save
25/25 [==============================] - 1s 6ms/step - loss: 5553877155840.0000
걸린시간 1.8949027061462402
loss :  5553877155840.0

MCP
25/25 [==============================] - 1s 7ms/step - loss: 5550363377664.0000
걸린시간 1.849578619003296
loss :  5550363377664.0
'''
