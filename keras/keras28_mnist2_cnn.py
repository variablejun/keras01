import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Conv2D에 들어가 연산하기 위해 3차원 데이터를 4차원 데이터로 쉐입을 바꾸어준다. 

# 0.992 이상으로만들기 , 캡쳐후 단톡에 전송
model = Sequential() 
model.add(Conv2D(50, kernel_size=(2,2) ,padding = 'same' ,input_shape=(28, 28, 1)))
model.add(Conv2D(4, (2,2), activation = 'relu'))
model.add(Conv2D(3, (2,2), activation = 'relu'))
model.add(Conv2D(2, (2,2), activation = 'relu'))

model.add(MaxPooling2D())
model.add(Flatten())#(N, 180)
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation='sigmoid')) 

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=20, mode='max', verbose=3)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.003, verbose=1,callbacks=[es]) 
loss = model.evaluate(x_test, y_test) 

print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''

??
loss :  0.0
accuracy :  0.11349999904632568

loss :  0.0
accuracy :  0.11349999904632568

'''
'''
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)
'''