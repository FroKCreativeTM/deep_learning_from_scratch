#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


x = np.array([1.0,2.0,3.0])
print(x)

type(x)


# In[7]:


A = np.array([[1.0,2.0],[3.0,4.0]])
print(A)
A.shape


# In[11]:


a = np.arange(0, 6 ,0.1)
b = np.sin(a)
c = np.cos(a)

plt.plot(a, b, label="sin")
plt.plot(a, c, linestyle="--" ,label="cos")
plt.legend()
plt.show()

