#!/usr/bin/env python
# coding: utf-8

# In[1]:


#path
import os
from os.path import isdir, join
from pathlib import Path

# Scientific Math 
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.offline as py
import plotly.graph_objs as go

#Deep learning
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

import random
import copy
import librosa

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print(os.listdir("D:/Users/26030219/Desktop/input"))  

"""

comment
"""


# In[3]:


train_audio_path = 'D:/Users/26030219/Desktop/input/train/audio/'
print(os.listdir(train_audio_path))


# In[4]:


dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
dirs.sort()
print('Number of labels: ' + str(len(dirs[0:])))
print(dirs)


# In[5]:


all_wav = []
unknown_wav = []
label_all = []
label_value = {}
target_list = ['yes', 'no','right', 'left']
unknown_list = [d for d in dirs if d not in target_list and d != '_background_noise_' ]
print('target_list : ',end='')
print(target_list)
print('unknowns_list : ', end='')
print(unknown_list)
print('silence : _background_noise_')
i=0;
background = [f for f in os.listdir(join(train_audio_path, '_background_noise_')) if f.endswith('.wav')]
background_noise = []
for wav in background : 
    samples, sample_rate = librosa.load(join(join(train_audio_path,'_background_noise_'),wav), sr = 16000)
    samples = librosa.feature.mfcc(samples, sr=16000, n_mfcc=10, fmax=4000)

    background_noise.append(samples)

for direct in dirs[1:]:
    waves = [f for f in os.listdir(join(train_audio_path, direct)) if f.endswith('.wav')]
    label_value[direct] = i
    i = i + 1
    print(str(i)+":" +str(direct) + " ", end="")
    for wav in waves:
        samples, sample_rate = librosa.load(join(join(train_audio_path,direct),wav), sr = 16000) # 1.5s x 16000 = 24000
        samples = librosa.feature.mfcc(samples, sr=16000, n_mfcc=10, fmax=4000)
        
       # if len(samples) > 16000 : 
        #    samples = samples[:16000]
        
        #if len(samples) != 16000 : 
         #   continue
            
        if direct in unknown_list:
            unknown_wav.append(samples)
        else:
            label_all.append(direct)
            all_wav.append([samples, direct])
"""
1) we create a dataset from the train folder
2) we specify target list yes,no,left,right 
3) unknown list bed bird cat dog 
4) silence and noise factors
5) our samples from folder we take using librosa function and load the system like speech variables 
"""


# In[6]:


wav_all = np.reshape(np.delete(all_wav,1,1),(len(all_wav)))
label_all = [i for i in np.delete(all_wav,0,1).tolist()]
print(len(all_wav))
print(len(label_all))


# In[7]:


wav_all


# In[136]:


#Random pick start point
def get_one_noise(noise_num = 0):
    selected_noise = background_noise[noise_num]
    start_idx = random.randint(0, len(selected_noise)- 1 - 8000)
    return selected_noise[start_idx:(start_idx + 8000)]


# In[138]:


max_ratio = 0.1
noised_wav = []
augment = 1
delete_index = []
for i in range(augment):
    new_wav = []
    noise = get_one_noise(i)
    for i, s in enumerate(wav_all):
        if len(s) != 8000:
            delete_index.append(i)
            continue
        s = s + (max_ratio * noise)
        noised_wav.append(s)
np.delete(wav_all, delete_index)
np.delete(label_all, delete_index)


# In[139]:


wav_vals = np.array([x for x in wav_all])
label_vals = [x for x in label_all]
wav_vals.shape
"""
wav_vals equal numpy array values from wav_all 
the wav_vals variable is an array of dimention 8575 and times 8000.
wav_vals.shape print the shape
if you want you can do same with print(wav_vals.shape)
"""


# In[140]:


labels = copy.deepcopy(label_vals)
for _ in range(augment):
    label_vals = np.concatenate((label_vals, labels), axis = 0)
label_vals = label_vals.reshape(-1,1)


# In[141]:


#knowns audio random sampling
unknown = unknown_wav
np.random.shuffle(unknown_wav)
unknown = np.array(unknown)
unknown = unknown[:2000*(augment+1)]
unknown_label = np.array(['unknown' for _ in range(2000*(augment+1))])
unknown_label = unknown_label.reshape(2000*(augment+1),1)


# In[142]:


delete_index = []
for i,w in enumerate(unknown):
    if len(w) != 8000:
        delete_index.append(i)
unknown = np.delete(unknown, delete_index, axis=0)


# In[143]:


silence_wav = []
num_wav = (2000*(augment+1))//len(background_noise)
for i, _ in enumerate(background_noise):
    for _ in range((2000*(augment+1))//len(background_noise)):
        silence_wav.append(get_one_noise(i))
silence_wav = np.array(silence_wav)
silence_label = np.array(['silence' for _ in range(num_wav*len(background_noise))])
silence_label = silence_label.reshape(-1,1)
silence_wav.shape


# In[144]:


wav_vals    = np.reshape(wav_vals,    (-1, 8000))
noised_wav  = np.reshape(noised_wav,  (-1, 8000))
unknown       = np.reshape(unknown,   (-1, 8000))
silence_wav = np.reshape(silence_wav, (-1, 8000))


# In[145]:


print(wav_vals.shape)
print(noised_wav.shape)
print(unknown.shape)
print(silence_wav.shape)


# In[146]:


print(label_vals.shape)
print(unknown_label.shape)
print(silence_label.shape)


# In[147]:


wav_vals = np.concatenate((wav_vals, noised_wav), axis = 0)
wav_vals = np.concatenate((wav_vals, unknown), axis = 0)
wav_vals = np.concatenate((wav_vals, silence_wav), axis = 0)


# In[148]:


label_vals = np.concatenate((label_vals, unknown_label), axis = 0)
label_vals = np.concatenate((label_vals, silence_label), axis = 0)


# In[149]:


print(len(wav_vals))
print(len(label_vals))


# In[150]:


train_wav, test_wav, train_label, test_label = train_test_split(wav_vals, label_vals, 
                                                                    test_size=0.2,
                                                                    random_state = 1993,
                                                                   shuffle=True)


# In[161]:


lr = 0.001
#generations = 20000
#num_gens_to_wait = 250
epochs = 70 # 20116 train_wav files, 512 * steps = 20116, steps = 40
# and 5030 test_wav files 
batch_size = 512
drop_out_rate = 0.5
input_shape = (8000,1,1)

"""
 Paul note it and we will write meaning and duties of parameters @can
"""


# In[162]:


train_wav = train_wav.reshape(-1,8000,1,1)
test_wav = test_wav.reshape(-1,8000,1,1)


# In[153]:


print(target_list[0:4])
label_value = target_list[0:4]
label_value.append('unknown')
label_value.append('silence')
print(label_value)


# In[154]:


new_label_value = dict()
for i, l in enumerate(label_value):
    new_label_value[l] = i
label_value = new_label_value


# In[155]:


temp = []
for v in train_label:
    temp.append(label_value[v[0]])
train_label = np.array(temp)

temp = []
for v in test_label:
    temp.append(label_value[v[0]])
test_label = np.array(temp)
train_label = keras.utils.to_categorical(train_label, len(label_value))
test_label = keras.utils.to_categorical(test_label, len(label_value))


# In[163]:


print('Train_Wav Demension : ' + str(np.shape(train_wav)))


# In[164]:


print('Train_Label Demension : ' + str(np.shape(train_label)))


# In[165]:


print('Test_Wav Demension : ' + str(np.shape(test_wav)))


# In[166]:


print('Test_Label Demension : ' + str(np.shape(test_label)))


# In[167]:


print('Number Of Labels : ' + str(len(label_value)))
#print('Number Of Labels : ' + str((label_value)))


# In[168]:


input_tensor = Input(shape=(input_shape))

x = layers.Conv2D(8, (13, 1), padding='same', activation='relu', strides=1)(input_tensor)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(8, (11,1), padding='same', activation='relu', strides=1)(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(16, (9,1), padding='same', activation='relu', strides=1)(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(16, (7,1), padding='same', activation='relu', strides=1)(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(16, (5,1), padding='same', activation='relu', strides=1)(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(32, (3,1), padding='same', activation='relu', strides=1)(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(32, (3,1), padding='same', activation='relu', strides=1)(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(32, (3,1), padding='same', activation='relu', strides=1)(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(32, (3,1), padding='same', activation='relu', strides=1)(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
output_tensor = layers.Dense(len(label_value), activation='softmax')(x)

model = tf.keras.Model(input_tensor, output_tensor)

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(lr = lr),
             metrics=['accuracy'])


# In[169]:


model.summary()


# In[170]:


history = model.fit(train_wav, train_label, validation_data=[test_wav, test_label],
          batch_size=batch_size, 
          epochs=epochs,
          verbose=1)


# In[171]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[172]:


model.save('model_new2.h5')


# In[176]:


# Start from here
model = load_model('model_new2.h5')


# In[177]:


model.summary()


# In[212]:


#path = 'D:/Users/26030219/Desktop/leftcan.wav'
path = 'D:/Users/26030219/Desktop/exmpls/left_hmd.wav'

samples, sample_rate = librosa.load(path, sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)

#print(samples.shape)
#print('samples')
samples = samples[0:8000]
#print(samples)

newsample = np.expand_dims(samples, axis=0)
newsample = np.expand_dims(newsample, axis=2)
newsample = np.expand_dims(newsample, axis=2)

print(newsample.shape)

model.predict(newsample)


# In[71]:


#['yes', 'no', 'right', 'left']
#['yes', 'no', 'right', 'left', 'unknown', 'silence']
#([[5.5688840e-01, 1.8191251e-01, 4.0388595e-02, 2.1989197e-01,
#        9.1846019e-04, 7.2281210e-08]], dtype=float32)


# In[ ]:


#yes = 0.556 no = 0.18  right =0.04 left = 0.21  unknown = 0.00009184 silence = 0.000000007228


# In[ ]:




