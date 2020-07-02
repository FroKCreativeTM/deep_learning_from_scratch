#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pylab as plt


# In[6]:


# 계단 함수 - 퍼셉트론은 0, 1 중 하나만 흐른다.
def step_function(x) : 
    y = x > 0
    return y.astype(np.int)

#x = np.array([-1.0, 1.0 ,2.0])
#print(step_function(x))


# In[7]:


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# In[9]:


# 시그모이드(S자 모양) 함수 - 연속적인 실수가 흐른다.
def sigmoid(x) : 
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))


# In[10]:


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# In[21]:


#ReLU 함수 - 0 이하면 0을 출력하고 그 위면 그 입력을 그대로 출력한다.
def ReLU(x) : 
    return np.maximum(0, x)


# In[20]:


# 항등 함수 - 입력을 그대로 출력
def identity_function(x) : 
    return x


# In[22]:


# 3층 신경망
def init_network() : 
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network

def forward(network, x) :
    W1, W2, W3 = network['W1'], network['W2'] , network['W3']
    b1, b2, b3 = network['b1'], network['b2'] , network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y


# In[27]:


network = init_network()
x = np.array([[1.0, 0.5],[0.75, 0.25]])
y = forward(network, x)
print(y)


# In[29]:


# 소프트맥스 함수 - n 개의 출력층의 뉴런 수 중 k번째 출력
def softmax(a) : 
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y)

print(np.sum(y))

