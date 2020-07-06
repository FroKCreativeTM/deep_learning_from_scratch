#!/usr/bin/env python
# coding: utf-8

# In[28]:


import sys, os
sys.path.append(os.pardir)   # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle


# In[17]:


def img_show(img) : 
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# In[18]:


(x_train, t_train), (x_test, t_test) =     load_mnist(flatten=True, normalize=False)


# In[19]:


# 각 데이터 형상을 출력
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


# In[20]:


img = x_train[0]
label = t_train[0]
print(label)


# In[21]:


print(img.shape) # 1*28*28
img = img.reshape(28, 28) # 다시 이미지를 돌린다.
print(img.shape)


# In[22]:


img_show(img)


# In[31]:


# 신경망의 추론 처리

# 훈련 데이터 불러오기
def get_data() :
    (x_train, t_train), (x_test, t_test) =         load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

# 시그모이드(S자 모양) 함수 - 연속적인 실수가 흐른다.
def sigmoid(x) : 
    return 1 / (1 + np.exp(-x))

#ReLU 함수 - 0 이하면 0을 출력하고 그 위면 그 입력을 그대로 출력한다.
def ReLU(x) : 
    return np.maximum(0, x)

# 항등 함수 - 입력을 그대로 출력
def identity_function(x) : 
    return x

# 신경망 초기화
def init_network() : 
    with open("sample_weight.pkl", 'rb') as f : 
        network = pickle.load(f)
        
    return network

# 추론 (전에 봤던 3층 신경망과 비슷하다.)
def predict_with_sigmoid(network, x) : 
    W1, W2, W3 = network['W1'], network['W2'] , network['W3']
    b1, b2, b3 = network['b1'], network['b2'] , network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

def predict_with_relu(network, x) : 
    W1, W2, W3 = network['W1'], network['W2'] , network['W3']
    b1, b2, b3 = network['b1'], network['b2'] , network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = ReLU(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = ReLU(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y


# In[33]:


x, t = get_data()
network = init_network()

accurary_cnt = 0
for i in range(len(x)) :
    y = predict_with_sigmoid(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i] : 
        accurary_cnt += 1
        
print("accuracy : " + str(float(accurary_cnt) / len(x)))    


# In[34]:


x, t = get_data()
network = init_network()

accurary_cnt = 0
for i in range(len(x)) :
    y = predict_with_relu(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i] : 
        accurary_cnt += 1
        
print("accuracy : " + str(float(accurary_cnt) / len(x)))    

