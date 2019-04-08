#!/usr/bin/env python
# coding: utf-8

# In[60]:


import math
import cv2
import numpy as np
import skimage.data 


# In[61]:


import matplotlib.pyplot as plt
from PIL import Image


# In[62]:


def tanh(x):
    return np.tanh(x)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def gauss(siz, sig):
    x, y = np.mgrid[-siz//2 + 1:siz//2 + 1, -siz//2 + 1:siz//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sig**2)))
    return g/g.sum()


# In[63]:


#k test var
def conv(image1, filter1):
    size1 = filter1.shape[0]
    k = 0
    n = image1.shape[0]
    m = image1.shape[1]
    l = []
    for r in range(size1//2, n - size1//2):
        k = k+1
        tmp = []
        for c in range(size1//2, m - size1//2):
            k = k+2
            cur = image1[r-size1//2:r+size1//2+1, c-size1//2:c+size1//2+1]
#             print(cur.shape)
            tmp.append(np.sum(cur * filter1))
        l.append(tmp)
    l = np.asarray(l)
    return l
#     print(l)


# In[64]:


#l
def conv_layer(image1, filters):
    res = []
    l = 0
    for it in range(filters.shape[0]):
        cur_filter = filters[it]
        l = l+1
        if(len(cur_filter.shape) > 2):
            tmp = conv(image1[0], cur_filter[0])
            for i in range(1, cur_filter.shape[0]):
                l=l+2
                tmp += conv(image1[i], cur_filter[i])
        else:
            tmp = conv(image1, cur_filter)
        res.append(tmp)
    res = np.asarray(res)
    return res


# In[65]:


#o
def pooling(feature_map, size=2, stride=2):
    if(len(feature_map.shape) == 2):
        n, m = feature_map.shape
        l = []
        p = 0
        for r in range(0, n-size+1, stride):
            p = p+1
            tmp = []
            for c in range(0, m-size+1, stride):
                p = p+2
                cur = feature_map[r:r+size, c:c+size]
                tmp.append(np.max(cur))
            l.append(tmp)
        l = np.asarray(l)
        return l
    else:
        d, n, m = feature_map.shape
        res = []
        for it in range(d):
            l = []
            for r in range(0, n-size+1, stride):
                tmp = []
                for c in range(0, m-size+1, stride):
                    cur = feature_map[it, r:r+size, c:c+size]
                    tmp.append(np.max(cur))
                l.append(tmp)
            res.append(l)
        res = np.asarray(res)
        return res


# In[66]:


## READ IMAGE
#l
om = cv2.imread("./d1.jpeg")

img_gray = cv2.cvtColor(om, cv2.COLOR_BGR2GRAY)
l = 0
om = cv2.resize(img_gray, (32, 32))
om = np.asarray(om, dtype=np.uint8)
l = l+1
plt.imshow(om)
print(om.shape)


# ### CONVOLUTION LAYER 1 

# In[67]:


## CREATE FILTERS

filters = np.zeros((6, 5, 5))
#o
l = 0
for it in range(6):
    l = l+1
    filters[it, :, :] = gauss(5, it+1)

## EXECUTE GIVEN ARCHITECTURE

conv1 = conv_layer(om, filters)
conv1 = tanh(conv1)
rows = 2
cols = math.ceil(conv1.shape[0]/2)
fig=plt.figure(figsize=(20, 20))
l = 0
for it in range(1, conv1.shape[0]+1):
    l = l+1
    fig.add_subplot(rows, cols, it)
    plt.imshow(conv1[it-1])
print(conv1.shape)


# ### POOLING LAYER 1

# In[68]:


#s
pool1 = pooling(conv1, 2, 2)
rows = 2
cols = math.ceil(pool1.shape[0]/2)
s = 0
fig=plt.figure(figsize=(20, 20))
for it in range(1, pool1.shape[0]+1):
    s = s+1
    fig.add_subplot(rows, cols, it)
    plt.imshow(pool1[it-1])
print(pool1.shape)


# ### CONVOLUTION LAYER 2

# In[69]:


#l
ff = np.random.rand(16, pool1.shape[0], 5, 5) - 0.5
# print(ff.shape)

conv2 = conv_layer(pool1, ff)
conv2 = np.asarray(conv2)
# conv2 = tanh(conv2)
# print(conv2[0])
rows = 2
cols = math.ceil(conv2.shape[0]/2)
l = 0
fig=plt.figure(figsize=(20, 20))
for it in range(1, conv2.shape[0]+1):
    l = l+1
    fig.add_subplot(rows, cols, it)
    plt.imshow(conv2[it-1])
print(conv2.shape)


# ### POOLING LAYER 2

# In[70]:


#l
pool2 = pooling(conv2)
rows = 2
cols = math.ceil(pool2.shape[0]/2)
l=0
fig=plt.figure(figsize=(20, 20))
for it in range(1, pool2.shape[0]+1):
    l=l+1
    fig.add_subplot(rows, cols, it)
    plt.imshow(pool2[it-1])
print(pool2.shape)


# ### FULLY CONNECTED CONVOLUTION LAYER 1

# In[71]:


#l 
ff = np.random.rand(120, pool2.shape[0], 5, 5) - 0.5
l = 0
fc1 = conv_layer(pool2, ff)
fc1 = np.asarray(fc1)
rows = 10
cols = math.ceil(fc1.shape[0]/10 + 1)
fig=plt.figure(figsize=(50, 50))
for it in range(1, fc1.shape[0]+1):
    l = l + 1
    fig.add_subplot(rows, cols, it)
    plt.imshow(fc1[it-1])
print(fc1.shape)


# ### FULLY CONNECTED DENSE LAYER 2

# In[72]:


fc1 = fc1.reshape((120, 1))
w2 = np.random.rand(84, 120) - 0.5
b2 = np.random.rand(84, 1) - 0.5

fc2 = np.dot(w2, fc1) + b2
fc2 = tanh(fc2)
# fc2[fc2<0]=0
print(fc2.shape)
print(fc2)


# ## OUTPUT SOFTMAX LAYER

# In[73]:


w3 = np.random.rand(10, 84) - 0.5
b3 = np.random.rand(10, 1) - 0.5

out = np.dot(w3, fc2) + b3
out = softmax(out)
print(out)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




