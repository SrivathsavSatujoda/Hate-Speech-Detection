#!/usr/bin/env python
# coding: utf-8

# # LSTM
# 
# #### Shanmukha Srivathsav Satujoda - ss3203

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from sklearn import preprocessing
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
tf.logging.set_verbosity(tf.logging.ERROR)
import seaborn as sb
from keras.preprocessing.sequence import pad_sequences


# ### Data Preprocessing

# In[2]:


df_train = pd.read_excel('train.xlsx')
df_test = pd.read_excel('test.xlsx')


# In[3]:


#Removing Contractions
replace_list = {r"i'm": 'i am',
                r"'re": ' are',
                r"let’s": 'let us',
                r"'s":  ' is',
                r"'ve": ' have',
                r"can't": 'can not',
                r"cannot": 'can not',
                r"shan’t": 'shall not',
                r"n't": ' not',
                r"'d": ' would',
                r"'ll": ' will',
                r"'scuse": 'excuse',
                ',': ' ,',
                '.': ' .',
                '!': ' !',
                '?': ' ?',
                '\s+': ' '}
def clean_text(text):
    text = text.lower()
    for s in replace_list:
        text = text.replace(s, replace_list[s])
    text = ' '.join(text.split())
    return text


# In[4]:


#Applyng all the necessary preprocessing
X_test = df_test['tweet'].apply(lambda p: clean_text(p))
phrase_len_test = X_test.apply(lambda p: len(p.split(' ')))
max_phrase_len_test = phrase_len_test.max()
X_train = df_train['tweet'].apply(lambda p: clean_text(p))
phrase_len_train = X_train.apply(lambda p: len(p.split(' ')))
max_phrase_len = phrase_len_train.max()


# In[5]:


y_train = df_train['class']


# In[6]:


#Tokenizing training data
max_words = 10000
tokenizer = Tokenizer(
    num_words = max_words,
    filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~'
)
tokenizer.fit_on_texts(X_train)


# In[7]:


#Tokenizing test data
max_words = 10000
tokenizer = Tokenizer(
    num_words = max_words,
    filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~'
)
tokenizer.fit_on_texts(X_test)


# In[8]:


#Adding padding sequences to unify the length
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequences(X_train, maxlen = max_phrase_len)
X_test = pad_sequences(X_test, maxlen = max_phrase_len_test)
y_train = to_categorical(y_train)


# ### LSTM architecture

# In[9]:


model_lstm = Sequential()
model_lstm.add(Embedding(input_dim = max_words, 
                         output_dim = 32, input_length = max_phrase_len))
model_lstm.add(SpatialDropout1D(0.3))
model_lstm.add(LSTM(28, dropout = 0.3, recurrent_dropout = 0.3))
model_lstm.add(Dense(20, activation = 'relu'))
model_lstm.add(Dense(10, activation = 'relu'))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(2, activation = 'softmax'))
adam = keras.optimizers.Adam(lr=0.001, 
                             epsilon=1e-08, decay=9.5e-07)
model_lstm.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)


# In[10]:


history = model_lstm.fit(
    X_train,
    y_train,
    validation_split = 0.2,
    epochs = 10,
    batch_size = 256
)


# ### Metrics and Validation

# In[11]:


y_test = df_test['class']
y_test_num = y_test.to_numpy()
pred_porb = model_lstm.predict(X_test)
pred = model_lstm.predict_classes(X_test)
#Precision Score
from sklearn.metrics import precision_score
print("Precision Score: ", precision_score(y_test, pred))
#F1 Scores
from sklearn.metrics import f1_score
print("F1 Score: ",f1_score(y_test, pred))
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test,pred))


# In[13]:


import scikitplot as skplt
skplt.metrics.plot_roc_curve(y_test_num, pred_porb)
plt.show()


# ### Looking at the convergence plots, Loss and Accuracy

# In[14]:


plt.clf()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[15]:


plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'y', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




