#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')


# In[3]:


df1.head()


# In[4]:


df2.head()


# In[5]:


df1.isnull().sum()


# In[6]:


df2.isnull().sum()


# In[7]:


import nltk


# In[8]:


nltk.download('stopwords')


# In[9]:


nltk.download('punkt')


# In[10]:


nltk.download('wordnet')


# In[11]:


from nltk.corpus import stopwords
import nltk, os, re, string
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

df1['text']=df1['text'].apply(remove_stopwords)
df2['text']=df2['text'].apply(remove_stopwords)


# In[12]:


df1.head()


# In[13]:


df2.head()


# In[14]:


import nltk
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords

lemma = WordNetLemmatizer()
def process_text(text):
    text = re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])", " ",text.lower())
    words = nltk.word_tokenize(text)
    words = [lemma.lemmatize(word) for word in words if word not in set(stopwords.words("english"))]
    text = " ".join(words)
        
    return text

df1["text"] = df1["text"].apply(process_text)
df2["text"] = df2["text"].apply(process_text)


# In[15]:


import emoji

def cleanTweet(txt):
    txt = re.sub(r'#','',txt)
    txt = re.sub(r'RT : ','',txt)
    txt = re.sub(r'\n','',txt)
    # to remove emojis
    txt = re.sub(emoji.get_emoji_regexp(), r"", txt)
    txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',txt)
    txt = re.sub(r"https?://\S+|www\.\S+","",txt)
    txt = re.sub(r"<.*?>","",txt)
    return txt  


# In[16]:


df1["text"] = df1["text"].apply(cleanTweet)
df2["text"] = df2["text"].apply(cleanTweet)


# In[17]:


df1.head()


# In[18]:


df1["length"] = df1["text"].apply(len)


# In[19]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
sns.countplot(df1[df1["target"] == 1]["length"],ax = ax1).set(title = "disaster tweets")
sns.countplot(df1[df1["target"] == 0]["length"],ax = ax2).set(title = "Not disaster tweets")
plt.show()


# In[20]:


train_disaster = df1[df1["target"] == 1]
train_not_disaster  = df1[df1["target"] == 0]


# In[21]:


vectorizer = CountVectorizer()


# In[22]:


train_disaster_countvectorizer = vectorizer.fit_transform(df1['text'])


# In[23]:


train_disaster_countvectorizer.toarray()


# In[24]:


label = df1['target']
label


# In[25]:


df2.head()


# In[26]:


df2.isnull().sum()


# In[27]:


test_disaster_countvectorizer = vectorizer.transform(df2['text'])


# In[28]:


test_disaster_countvectorizer.toarray()


# In[29]:


from sklearn.svm import SVC


# In[30]:


model_svc=SVC(C=100,kernel='rbf')


# In[31]:


model_svc.fit(train_disaster_countvectorizer, label)


# In[32]:


test_sample = test_disaster_countvectorizer.toarray()
test_sample


# In[33]:


test_sample.shape


# In[34]:


prediction = model_svc.predict(test_sample)


# In[35]:


prediction_df = pd.DataFrame(prediction, columns=['target'])


# In[36]:


prediction_df


# In[37]:


predicted_result = pd.concat([df2['id'], prediction_df], axis=1)
predicted_result


# In[38]:


predicted_result.isnull().sum()


# In[39]:


predicted_result['target'].value_counts()


# In[40]:


predicted_result.to_csv('predicted_result123.csv', index=False)


# In[41]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_disaster_countvectorizer, label, test_size=0.2)


# In[42]:


model_svc.fit(x_train, y_train)


# In[43]:


predictions_svc = model_svc.predict(x_test)


# In[44]:


print(classification_report(y_test, predictions_svc))


# In[45]:


from sklearn.neighbors import KNeighborsClassifier
model_knc = KNeighborsClassifier()
model_knc.fit(x_train, y_train)


# In[46]:


predictions_knc = model_knc.predict(x_test)


# In[47]:


print(classification_report(y_test, predictions_knc))


# In[ ]:





# In[48]:


sns.countplot(df1['target'])


# In[ ]:




