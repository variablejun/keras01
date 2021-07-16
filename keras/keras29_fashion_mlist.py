# 이미지 확인
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(x_train.shape,y_train.shape )
print(x_test.shape,y_test.shape )
plt.imshow(x_train[11], 'gray') # 뭐 이상한거 나옴
plt.show()
'''

'''