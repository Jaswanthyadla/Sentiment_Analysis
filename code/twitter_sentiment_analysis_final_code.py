#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re # for regular expressions
import pandas as pd 
pd.set_option("display.max_colwidth", 200)
import os
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk # for text manipulation
import warnings 
import wordcloud
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


pwd


# In[4]:


train_set= pd.read_csv(r'trainset copy final.csv')
test_set= pd.read_csv(r'tst2.csv')


# In[5]:


test_set.head()
train_set.shape 


# In[6]:


#displaying some non-negative tweets
train_set[train_set['label'] == 0].head(10) 


# In[7]:


#displaying some negative tweets
train_set[train_set['label'] == 1].head(10) 


# In[8]:


#distribution of length of length of both train and test tweets
length_train_set = train_set['tweet'].str.len()
length_test_set = test_set['tweet'].str.len()

plt.hist(length_train_set, bins=20, label="train_Set_tweets")
plt.hist(length_test_set, bins=20, label="test_Set_tweets")
plt.legend()
plt.show()


# In[9]:


#combining both train and test tweets
combine = train_set.append(test_set, ignore_index=True,sort=False)
combine.shape 


# In[10]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt


# In[11]:


combine['cleaned_tweet'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*") 
combine.head()


# In[12]:


combine['cleaned_tweet'] = combine['cleaned_tweet'].str.replace("[^a-zA-Z#]", " ")
combine.head(10)


# In[13]:


combine['cleaned_tweet'] = combine['cleaned_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
combine.head()


# In[14]:


tokenized_tweet = combine['cleaned_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()


# In[15]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 


# In[16]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    
combine['cleaned_tweet'] = tokenized_tweet 
combine.head()


# In[17]:


all_words = ' '.join([text for text in combine['cleaned_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white',width=900, height=450, random_state=21, max_font_size=100).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[18]:


normal_words =' '.join([text for text in combine['cleaned_tweet'][combine['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[19]:


negative_words = ' '.join([text for text in combine['cleaned_tweet'][combine['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[20]:


#Defining a function to collect hashtags
def extract_hashtag(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[21]:


#extracting hashtags from non-negative tweets
HT_regular = extract_hashtag(combine['cleaned_tweet'][combine['label'] == 0])

# extracting hashtags from negative tweets
HT_negative = extract_hashtag(combine['cleaned_tweet'][combine['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


# In[22]:


a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags     
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[23]:


b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 20 most frequent hashtags
e = e.nlargest(columns="Count", n = 20)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")


# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim


# In[25]:


bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combine['cleaned_tweet'])
bow.shape 


# In[26]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combine['cleaned_tweet'])
tfidf.shape


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# In[28]:



train_bow = bow[:31949,:]
test_bow = bow[31949:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_set['label'],  
                                                          random_state=42, 
                                                          test_size=0.3)


# In[29]:


lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 then 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1 score 


# In[30]:


test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test_set['label'] = test_pred_int
submission = test_set[['id','label','tweet']]
submission.to_csv('sub_lreg_bow1.csv', index=False)
submission.shape 


# In[31]:



train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index] 


# In[32]:


lreg=LogisticRegression()
lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) 


# In[ ]:




