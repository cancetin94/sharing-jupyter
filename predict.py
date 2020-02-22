#!/usr/bin/env python
# coding: utf-8

# In[1]:


#path
import os

# Scientific Math 
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import tensorflow as tf

#Deep learning
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

import librosa


# In[2]:


model = load_model('seventy_light_nxp.h5')


# In[3]:


model.summary()


# In[4]:


# path = 'D:/Users/26030219/Desktop/yes_hmd.wav'
path = 'D:/Users/26030219/Desktop/exmpls/no_hmd.wav'

samples, sample_rate = librosa.load(path, sr = 16000)

print(samples.shape)
print(samples)

samples = samples[0:16000]


# In[5]:


newsample = np.expand_dims(samples, axis=0)
newsample = np.expand_dims(newsample, axis=2)
newsample = np.expand_dims(newsample, axis=2)

print(newsample.shape)


# In[6]:


model.predict(newsample)

#{'yes': 0, 'no': 1, 'on': 2, 'off': 3, 'stop': 4, 'go': 5, 'unknown': 6} 


# In[ ]:




