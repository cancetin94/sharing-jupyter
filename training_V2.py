#!/usr/bin/env python
# coding: utf-8

# In[1]:


#path
import os

# Scientific Math 
import numpy as np

from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import tensorflow as tf

#Deep learning
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

import random
import copy
import librosa


# In[2]:


train_audio_path = 'D:/Users/26030219/Desktop/input/train/audio/'
print(os.listdir(train_audio_path))

os.listdir(train_audio_path)
dirs = [f for f in os.listdir(train_audio_path) if os.path.isdir(train_audio_path + '/' + f)]
dirs.sort()
print('Number of labels: ' + str(len(dirs[0:])))
print(dirs)


# In[3]:


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
background = [f for f in os.listdir(train_audio_path + '/' + '_background_noise_') if f.endswith('.wav')]
background_noise = []
for wav in background : 
    samples, sample_rate = librosa.load(train_audio_path + '/' + '_background_noise_' + '/' + wav, sr = 16000)
    background_noise.append(samples)

for direct in dirs[1:]:
    waves = [f for f in os.listdir(train_audio_path + '/' + direct) if f.endswith('.wav')]
    label_value[direct] = i
    i = i + 1
    print(str(i)+":" +str(direct) + " ", end="")
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + direct + '/' + wav, sr = 16000)
        if len(samples) != 16000 : 
            continue
            
        if direct in unknown_list:
            unknown_wav.append(samples)
        else:
            label_all.append(direct)
            all_wav.append([samples, direct])


# In[1]:


print(len(all_wav))
print(len(label_all))


# In[29]:


wav_all = np.reshape(np.delete(all_wav,1,1),(len(all_wav)))
label_all = [i for i in np.delete(all_wav,0,1).tolist()]

#Random pick start point
def get_one_noise(noise_num = 0):
    selected_noise = background_noise[noise_num]
    start_idx = random.randint(0, len(selected_noise)- 1 - 16000)
    return selected_noise[start_idx:(start_idx + 16000)]

max_ratio = 0.1
noised_wav = []
augment = 1
delete_index = []
for i in range(augment):
    new_wav = []
    noise = get_one_noise(i)
    for i, s in enumerate(wav_all):
        if len(s) != 16000:
            delete_index.append(i)
            continue
        s = s + (max_ratio * noise)
        noised_wav.append(s)
np.delete(wav_all, delete_index)
np.delete(label_all, delete_index)


# In[30]:


wav_vals = np.array([x for x in wav_all])
label_vals = [x for x in label_all]
wav_vals.shape

labels = copy.deepcopy(label_vals)
for _ in range(augment):
    label_vals = np.concatenate((label_vals, labels), axis = 0)
label_vals = label_vals.reshape(-1,1)


# In[31]:


#knowns audio random sampling
unknown = unknown_wav
np.random.shuffle(unknown_wav)
unknown = np.array(unknown)
unknown = unknown[:2000*(augment+1)]
unknown_label = np.array(['unknown' for _ in range(2000*(augment+1))])
unknown_label = unknown_label.reshape(2000*(augment+1),1)

delete_index = []
for i,w in enumerate(unknown):
    if len(w) != 16000:
        delete_index.append(i)
unknown = np.delete(unknown, delete_index, axis=0)

silence_wav = []
num_wav = (2000*(augment+1))//len(background_noise)
for i, _ in enumerate(background_noise):
    for _ in range((2000*(augment+1))//len(background_noise)):
        silence_wav.append(get_one_noise(i))
silence_wav = np.array(silence_wav)
silence_label = np.array(['silence' for _ in range(num_wav*len(background_noise))])
silence_label = silence_label.reshape(-1,1)
print(silence_wav.shape)


# In[32]:


wav_vals    = np.reshape(wav_vals,    (-1, 16000))
noised_wav  = np.reshape(noised_wav,  (-1, 16000))
unknown       = np.reshape(unknown,   (-1, 16000))
silence_wav = np.reshape(silence_wav, (-1, 16000))

print(wav_vals.shape)
print(noised_wav.shape)
print(unknown.shape)
print(silence_wav.shape)

print(label_vals.shape)
print(unknown_label.shape)
print(silence_label.shape)


# In[33]:


wav_vals = np.concatenate((wav_vals, noised_wav), axis = 0)
wav_vals = np.concatenate((wav_vals, unknown), axis = 0)
wav_vals = np.concatenate((wav_vals, silence_wav), axis = 0)

label_vals = np.concatenate((label_vals, unknown_label), axis = 0)
label_vals = np.concatenate((label_vals, silence_label), axis = 0)

print(len(wav_vals))
print(len(label_vals))


# In[34]:


train_wav, validation_wav, train_label, validation_label = train_test_split(wav_vals, label_vals, 
                                                                    test_size=0.2,
                                                                    shuffle=True)


# In[35]:


lr = 0.001
epochs = 70 # 20116 train_wav files, 512 * steps = 20116, steps = 40
# and 5030 test_wav files 
batch_size = 512
drop_out_rate = 0.5
input_shape = (16000,1)


# In[36]:


train_wav = train_wav.reshape(-1,16000,1,1)
validation_wav = validation_wav.reshape(-1,16000,1,1)

print(target_list[0:4])
label_value = target_list[0:4]
label_value.append('unknown')
label_value.append('silence')
print(label_value)

new_label_value = dict()
for i, l in enumerate(label_value):
    new_label_value[l] = i
label_value = new_label_value


# In[37]:


temp = []
for v in train_label:
    temp.append(label_value[v[0]])
train_label = np.array(temp)

temp = []
for v in validation_label:
    temp.append(label_value[v[0]])
validation_label = np.array(temp)
train_label = keras.utils.to_categorical(train_label, len(label_value))
validation_label = keras.utils.to_categorical(validation_label, len(label_value))

print('Train_Wav Demension : ' + str(np.shape(train_wav)))
print('Train_Label Demension : ' + str(np.shape(train_label)))
print('validation_wav Demension : ' + str(np.shape(validation_wav)))
print('validation_label Demension : ' + str(np.shape(validation_label)))

print('Number Of Labels : ' + str(len(label_value)))
#print('Number Of Labels : ' + str((label_value)))


# In[31]:


input_tensor = Input(shape=(input_shape))

x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_tensor)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
output_tensor = layers.Dense(len(label_value), activation='softmax')(x)

model = tf.keras.Model(input_tensor, output_tensor)

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(lr = lr),
             metrics=['accuracy'])


# In[97]:


model.summary()


# In[32]:


history = model.fit(train_wav, train_label, validation_data=[test_wav, test_label],
          batch_size=batch_size, 
          epochs=epochs,
          verbose=1)


# In[35]:


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


# In[36]:


model.save('model_new.h5')


# In[6]:


model = load_model('seventy_light_nxp.h5')


# In[7]:


model.summary()


# In[15]:


# path = 'D:/Users/26030219/Desktop/leftcan.wav'
path = 'D:/Users/26030219/Desktop/exmpls/go_hmd.wav'

samples, sample_rate = librosa.load(path, sr = 16000)
#samples = librosa.resample(samples, sample_rate, 8000)

print(samples.shape)
print(samples)

samples = samples[0:16000]


# In[16]:


newsample = np.expand_dims(samples, axis=0)
newsample = np.expand_dims(newsample, axis=2)
newsample = np.expand_dims(newsample, axis=2)

print(newsample.shape)


# In[18]:


model.predict(newsample)


# In[ ]:


#['yes', 'no', 'right', 'left']
#['yes', 'no', 'right', 'left', 'unknown', 'silence']
#([[5.5688840e-01, 1.8191251e-01, 4.0388595e-02, 2.1989197e-01,
#        9.1846019e-04, 7.2281210e-08]], dtype=float32)


# In[ ]:


#yes = 0.556 no = 0.18  right =0.04 left = 0.21  unknown = 0.00009184 silence = 0.000000007228


# In[ ]:




