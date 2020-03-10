#!/usr/bin/env python
# coding: utf-8

# # Feed Forward Neural Nets
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
from keras.utils import to_categorical
from keras.layers import Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from keras.callbacks import TensorBoard
from sklearn import preprocessing
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
tf.logging.set_verbosity(tf.logging.ERROR)
import seaborn as sb


# In[2]:


#Reading Data
df = pd.read_excel("data.xlsx")
tweets = df.tweet


# In[3]:


#Librarires required for text processing
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re


# In[4]:


stopwords=stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()


# In[5]:


"""Funtions written to be passed as arguments"""
def pre_process(strs):
    #remove special chars
    strs = re.sub(r'[?|$|.|!]',r'',strs)
    #remove everything except for alpha numeric values
    strs = re.sub(r'[^a-zA-Z0-9 ]',r'',strs) 
    #to remove numbers
    strs=''.join(c if c not in map(str,range(0,10)) else '' for c in strs) 
    #remove extra spaces
    strs = re.sub('  ',' ',strs) 
    strs = strs.lower()
    return strs

def tokenize_stem(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


vectorizer = TfidfVectorizer(
    tokenizer=tokenize_stem,
    ngram_range=(1, 3),
    preprocessor = pre_process,
    stop_words=stopwords, 
    use_idf=True,
    smooth_idf=False,
    norm=None, 
    decode_error='replace',
    max_features=200,
    min_df=5,
    max_df=0.501
    )


# In[6]:


tfidf = vectorizer.fit_transform(tweets).toarray()
vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
idf_vals = vectorizer.idf_
idf_dict = {i:idf_vals[i] for i in vocab.values()} #keys are indices; values are IDF scores


# In[7]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *

sentiment_analyzer = VS()

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    ##SENTIMENT
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    
    words = pre_process(tweet) #Get text only
    #count syllables in words
    syllables = textstat.syllable_count(words)
    #num chars in words
    num_chars = sum(len(w) for w in words) 
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + 
                 float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) -
                (84.6*float(avg_syl)),2)
    
    twitter_objs = count_twitter_objs(tweet) #Count #, @, and http://
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE,syllables, avg_syl, num_chars, 
                num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], 
                sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    #features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


# In[8]:


other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word",
                        "num_chars", "num_chars_total", \
                        "num_terms", "num_words", 
                        "num_unique_words", "vader neg",
                        "vader pos","vader neu", "vader compound", \
                        "num_hashtags", "num_mentions", "num_urls", "is_retweet"]


# In[9]:


feats = get_feature_array(tweets)


# In[10]:


#Now join them all up
M = np.concatenate([tfidf,feats],axis=1)


# In[11]:


#Finally get a list of variable names
variables = ['']*len(vocab)
for k,v in vocab.items():
    variables[v] = k
feature_names = variables+other_features_names


# In[12]:


X = pd.DataFrame(M)
y = df['class'].astype(int)


# In[13]:


plt.hist(y)


# In[14]:


#splitting the data to train and test datasets
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 23)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# ### Baseline Logistic Regression

# In[15]:


from sklearn.linear_model import LogisticRegression as LR
clf = LR(random_state=0).fit(X_train, y_train)
pred_lr = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy: ",accuracy_score(y_test, pred_lr))
y_test_num = y_test.to_numpy()
pred_prob_lr = clf.predict_proba(X_test)


# ### ROC curve and F1 Score

# In[16]:


y_test_num = y_test.to_numpy()
pred_prob_lr = clf.predict_proba(X_test)
import scikitplot as skplt
skplt.metrics.plot_roc_curve(y_test_num, pred_prob_lr)
plt.show()


# In[17]:


from sklearn.metrics import f1_score
print("F1 Score: ", f1_score(y_test, pred_lr))


# ### Feed forward neural network

# In[18]:


NN_model = Sequential()
    # The Input Layer :
NN_model.add(Dense(64, kernel_initializer='normal',
                   input_dim=X_train.shape[1], activation='relu'))

#Hidden Layers
NN_model.add(Dense(32, kernel_initializer='normal', 
                   activation='relu'))
NN_model.add(Dense(16, kernel_initializer='normal', 
                   activation='relu'))
NN_model.add(Dense(8, kernel_initializer='normal', 
                   activation='relu'))
#Output Layer
NN_model.add(Dense(2, kernel_initializer='normal',
                   activation='softmax'))



adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, 
                             epsilon=1e-08, decay=9.5e-07)


NN_model.compile(loss='binary_crossentropy', 
                 optimizer=adam, metrics=['acc'])
NN_model.summary()


# In[19]:


#Fitting the model
NN_model.fit(X_train, y_train_cat, epochs=20, batch_size=16, validation_split = 0.2)


# ### Validation metrics

# In[20]:


#Roc Curve
prediction_prob = NN_model.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test_num, prediction_prob)


# In[24]:


#F1 score and accuracy
preds_nn = NN_model.predict_classes(X_test)
print("F1 Score: ", f1_score(y_test, preds_nn))
print("Accuracy:", accuracy_score(y_test,preds_nn))


# ### Looking at the convergence plots, Loss and Accuracy

# In[22]:


plt.clf()
loss = NN_model.history.history['loss']
val_loss = NN_model.history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[23]:


plt.clf()
acc = NN_model.history.history['acc']
val_acc = NN_model.history.history['val_acc']
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'y', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

