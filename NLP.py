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


# In[ ]:





# In[5]:


train_disater = df1[df1['target'] == 1]


# In[6]:


train_disater


# In[7]:


train_not_disater = df1[df1['target'] == 0]


# In[8]:


train_not_disater


# In[9]:


vectorizer = CountVectorizer()


# In[10]:


train_disaster_countvectorizer = vectorizer.fit_transform(df1['text'])


# In[11]:


train_disaster_countvectorizer.toarray()


# In[12]:


label = df1['target']
label


# In[13]:


df2.head()


# In[14]:


df2.isnull().sum()


# In[15]:


test_disaster_countvectorizer = vectorizer.transform(df2['text'])


# In[16]:


test_disaster_countvectorizer.toarray()


# # Training with train data and evaluation with given test data

# ## Support Vector Machine

# In[17]:


from sklearn.svm import SVC


# In[18]:


model_svc=SVC(C=100,kernel='rbf')


# In[19]:


model_svc.fit(train_disaster_countvectorizer, label)


# In[20]:


test_sample = test_disaster_countvectorizer.toarray()
test_sample


# In[21]:


test_sample.shape


# In[22]:


prediction = model_svc.predict(test_sample)


# In[23]:


prediction_df = pd.DataFrame(prediction, columns=['target'])


# In[24]:


prediction_df


# In[25]:


predicted_result = pd.concat([df2['id'], prediction_df], axis=1)
predicted_result


# In[26]:


predicted_result.isnull().sum()


# In[27]:


predicted_result['target'].value_counts()


# In[28]:


predicted_result.to_csv('predicted_result.csv', index=False)


# # Training and valuation with splitted train test data

# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_disaster_countvectorizer, label, test_size=0.2)


# ## Support Vector Machine Classifier

# In[30]:


model_svc.fit(x_train, y_train)


# In[31]:


predictions_svc = model_svc.predict(x_test)


# In[32]:


print(classification_report(y_test, predictions_svc))


# ## KNeighborsClassifier

# In[33]:


from sklearn.neighbors import KNeighborsClassifier
model_knc = KNeighborsClassifier()
model_knc.fit(x_train, y_train)


# In[34]:


predictions_knc = model_knc.predict(x_test)


# In[35]:


print(classification_report(y_test, predictions_knc))


# ## Random Forest Classifier

# In[36]:


from sklearn.ensemble import RandomForestClassifier
model_rfc = RandomForestClassifier()
model_rfc.fit(x_train, y_train)


# In[37]:


predictions_rfc = model_rfc.predict(x_test)


# In[38]:


print(classification_report(y_test, predictions_rfc))


# In[ ]:





# ## Multinomial Naive Bayes

# In[39]:


from sklearn.naive_bayes import MultinomialNB

model_mnb = MultinomialNB()
model_mnb.fit(x_train, y_train)


# In[40]:


predictions_mnb = model_mnb.predict(x_test)


# In[41]:


print(classification_report(y_test, predictions_mnb))


# In[ ]:




