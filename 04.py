#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# 손실함수는 신경망 성능의 나쁨을 나타내는 지표이다.
# 잘 처리하지 못함을 나타낸다.

# 오차 제곱합은 가장 많이 쓰이는 손실 함수이다.
# @Param
# y : 신경망의 출력값
# t : 정답 레이블
def sum_squares_error(y, t) : 
    return 0.5 * np.sum((y - t) ** 2)

# 교차 엔트로피 오차
# 여기서는 자연 로그를 이용해서 성능을 구한다.
# 즉 정답에 해당하는 출력이 커질수록 오류값을 0에 다가가고
# 반대로 정답일 때의 출력이 작아질수록 오차는 커진다.
def cross_entropy_error(y, t) : 
    delta = 1e-7   # -inf 즉 언더플로우 방지
    return -np.sum(t * np.log(y + delta))    


# In[3]:


# 미니 배치 연습 (훈련 데이터의 일부만 가져와서 훈련하는 것)
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) =     load_mnist(normalize=True, one_hot_label=True) # one_hot_label : 정답 위치가 1

# 훈련 데이터 6만개중 784개만 입력한다.
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)


# In[5]:


# 훈련 데이터에서 무작위로 10개를 가져오는 방법
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 무작위로 배열에 원소 10개를 가져온다.
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# In[7]:


# 배치용 교차 엔트로피 오차 구현
# 미니배치 같은 배치 데이터를 지원하는 교차 엔트로피 오차를 구현한다.
def cross_entropy_error(y, t) :
    if y.dim == 1 : 
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    # 이 구현은 원핫 인코딩일 때 t가 0인 원소는 오차도 0이기에 무시해도 좋다.
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# In[15]:


# 여기서 중요한 점은 정확도가 아닌 손실의 크기를 찾는 이유는 바로
# 미분한 경우 정확도를 기준으로 하면 대부분의 장소에서 0이 되기 때문이다.

# 본인은 성능 때문에 시그모이드와 계단 함수를 따로 쓴다고 생각했으나,
# 사실 계단 함수는 대부분의 장소에서 기울기가 0이지만, 시그모이드 함수는
# 함수의 기울기가 즉 접선이 0이 아니기 때문이다.

# 그럼 이제 미분을 구해보자
#def numerical_diff(f, x) : 
#    h = 10e-50 
#    return (f(x + h) - f(x)) / h

# 위의 예시는 나쁜 예시이다.
# 가급적 작은 값을 대입하고 싶었기에 저런 식으로 h를 구현했지만
# 반올림 오차가 발생한다는 문제가 있다.

# 두번째는 함수 f의 차분(임의 두 점에서의 함수 값들의 차이)과 관련된 점이다.
# x + h와 x 사이의 함수 f의 차분을 계산하고 싶지만 이 계산은 오차가 있다.
# 진정한 미분은 x위치의 함수의 기울기에 해당하지만, 이 구현은 (x+h) 와 x
# 사이의 기울기이다. 즉 h를 무한히 0으로 좁히는 것이 불가능해서 생기는 한계이다.
# 이 오차를 줄이기 위해 (x+h)와 (x-h)일 떄의 함수 f의 차분을 계산할 필요가 있다.
# 이를 중심 차분이라고 한다.
def numerical_diff(f, x) : 
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 위처럼 아주 작은 차분으로 미분하는 것을 수치 미분이라고 한다.


# In[10]:


# 간단한 함수 1
def function_1(x) : 
    return 0.01*x**2 + 0.1*x


# In[11]:


import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x) 
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, y)


# In[17]:


# f(x)의 변화량
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))


# In[18]:


# 입력이 두개 이상을 미분하는 편미분
def function_2 (x) : 
    return x[0] ** 2 + x[1] ** 2


# In[19]:


# 실험용
def function_tmp1 (x0) : 
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))


# In[20]:


def function_tmp2(x1) : 
    return 3.0**2.0 + x1*x1

print(numerical_diff(function_tmp2, 4.0))


# In[21]:


# 기울기
# 모든 변수의 편미분을 벡터로 정리한 것을 기울기(gradient)라고 한다.
# 동작 방식은 변수가 하나 뿐인 미분과 동일하다.
def numerical_gradient(f, x) : 
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size) : 
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad


# In[22]:


# 편미분 기울기 결과
# 기울기는 각 지점에서 낮아지는 방향을 가리킨다.
# 즉 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향이다.
print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))


# In[25]:


# 경사(하강)법
# 기계학습 문제 대부분은 학습 단계에서 최적의 매개 변수를 찾아낸다.
# 신경망 역시 최적의 매개변수(가중치와 편향)을 학습 시에 찾아야한다.
# 그러나 손실함수는 매우 복잡하다.
# 기울기를 잘 이용해서 함수의 최솟값을 찾아보자. (안정점이라는 곳은 기울기가 0이기에 여길 찾아야..)
# 이런 식으로 기울기를 잘 찾아서 함수의 값을 줄이는 것을 바로 경사법이라 한다.

# @Param
# f : 특정한 함수
# init_x : 초깃값
# lr : learning rate 학습률
# step_num : 경사법에 따른 반복 횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100) : 
    x = init_x 
    
    for i in range(step_num) : 
        grad = numerical_gradient(f, x) 
        x -= lr * grad
        
    return x


# In[26]:


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))


# In[ ]:




