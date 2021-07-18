#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix


# # Taking Data

# In[2]:


df1= pd.read_csv('train.csv')
df2=pd.read_csv('test.csv')


# # Preprocessing and Data Visualization

# In[3]:


df1.head()


# In[4]:


df1.isnull().sum()


# In[5]:


df1["length"] = df1["text"].apply(len)


# In[6]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
sns.distplot(df1[df1["target"] == 1]["length"],ax = ax1,label='Disater Tweets',color='blue').set(title = "disaster tweets")
sns.distplot(df1[df1["target"] == 0]["length"],ax = ax2,label='Non-Disater Tweets',color='green').set(title = "Not disaster tweets")
fig.suptitle('Tweet length distribution')
fig.legend(loc='upper right')
plt.show()


# In[7]:


df1['words_count'] = df1['text'].str.split().map(lambda x: len(x))
df1.head()


# In[8]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
dis_tweet=df1[df1['target']==1]['words_count']
ax1.hist(dis_tweet,color='blue')
ax1.set_title('Disaster tweets')
ax1.grid()
nondis_tweet=df1[df1['target']==0]['words_count']
ax2.hist(nondis_tweet,color='green')
ax2.set_title('Non-disaster tweets')
ax2.grid()
fig.suptitle('Words in a tweet')
plt.show()


# In[ ]:





# In[9]:


train_disaster = df1[df1['target'] == 1]


# In[10]:


train_disaster


# In[11]:


train_not_disater = df1[df1['target'] == 0]


# In[12]:


train_not_disater


# In[13]:


sns.countplot(df1['target'])


# In[14]:


vectorizer = CountVectorizer()


# In[15]:


train_disaster_countvectorizer = vectorizer.fit_transform(df1['text'])


# In[16]:


train_disaster_countvectorizer.toarray()


# In[17]:


label = df1['target']
label


# In[18]:


df2.head()


# In[19]:


df2.isnull().sum()


# In[20]:


test_disaster_countvectorizer = vectorizer.transform(df2['text'])


# In[21]:


test_disaster_countvectorizer.toarray()


# # Training with train data and evaluation with given test data

# ## Support Vector Machine

# In[22]:


from sklearn.svm import SVC


# In[23]:


model_svc=SVC(C=100,kernel='rbf')


# In[24]:


model_svc.fit(train_disaster_countvectorizer, label)


# In[25]:


test_sample = test_disaster_countvectorizer.toarray()
test_sample


# In[26]:


test_sample.shape


# In[27]:


prediction = model_svc.predict(test_sample)


# In[28]:


prediction_df = pd.DataFrame(prediction, columns=['target'])


# In[29]:


prediction_df


# In[30]:


predicted_result = pd.concat([df2['id'], prediction_df], axis=1)
predicted_result


# In[31]:


predicted_result.isnull().sum()


# In[32]:


predicted_result['target'].value_counts()


# In[33]:


predicted_result.to_csv('predicted_result.csv', index=False)


# # Training and valuation with splitted train test data

# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_disaster_countvectorizer, label, test_size=0.2)


# ## Support Vector Machine Classifier

# In[35]:


model_svc.fit(x_train, y_train)


# In[36]:


predictions_svc = model_svc.predict(x_test)


# In[37]:


print(classification_report(y_test, predictions_svc))


# ## KNeighborsClassifier

# In[38]:


from sklearn.neighbors import KNeighborsClassifier
model_knc = KNeighborsClassifier()
model_knc.fit(x_train, y_train)


# In[39]:


predictions_knc = model_knc.predict(x_test)


# In[40]:


print(classification_report(y_test, predictions_knc))


# ## Random Forest Classifier

# In[41]:


from sklearn.ensemble import RandomForestClassifier
model_rfc = RandomForestClassifier()
model_rfc.fit(x_train, y_train)


# In[42]:


predictions_rfc = model_rfc.predict(x_test)


# In[43]:


print(classification_report(y_test, predictions_rfc))


# In[ ]:





# ## Multinomial Naive Bayes

# In[44]:


from sklearn.naive_bayes import MultinomialNB

model_mnb = MultinomialNB()
model_mnb.fit(x_train, y_train)


# In[45]:


predictions_mnb = model_mnb.predict(x_test)


# In[46]:


print(classification_report(y_test, predictions_mnb))


# In[ ]:





# In[ ]:




