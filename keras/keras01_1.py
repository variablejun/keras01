
from tensorflow.keras.models import Sequential   # 케라스는 순차적 모델 과 함수형 모델 존재
from tensorflow.keras.layers import Dense
import numpy as np #

#1.data
x = np.array([1,2,3]) # 스칼라3개 벡터1 1차원
y = np.array([1,2,3]) # 스칼라3개 벡터1 1차원

#2.모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))  #  앞에 output 1-> y , input_dim=1 1차원, input_dim=1 -> x

#3.complie/훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, epochs=10000, batch_size=1) #  epoch 훈련횟수/ batch 1 2 3 을 하나씩할지 한꺼번에 할지(batch_size=3)
# 훈련을 많이 한다고 좋지않다. 조절을 해야한다.(Hyper parameter 튜닝)
# 훈련할때마다 가중치가 바뀐다. 그래서 최적의 가중치를 찾으면 저장한다.
#평가/예측
loss = model.evaluate(x, y) 

print('loss : ', loss)

result = model.predict([4]) # fit에서 생성된 w과 b가 들어가있다.

print('4의 예측값 : ', result)
