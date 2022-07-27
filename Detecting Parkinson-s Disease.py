#!/usr/bin/env python
# coding: utf-8

# # Detecting Parkinson's Disease

# In[1]:


#importing

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[2]:


#converting dataset to .csv format

dataset=pd.read_csv(r'C:\Users\Aditi\Desktop\Projects\Parkinson-s Disease\parkinsons.data') 
dataset.to_csv (r'C:\Users\Aditi\Desktop\Projects\Parkinson-s Disease\parkinsons.csv', index=None)


# In[3]:


dataset.shape


# In[4]:


dataset.head()


# In[5]:


dataset.info()


# In[6]:


dataset.isnull().sum()


# In[7]:


dataset.describe()


# In[8]:


dataset['status'].value_counts()


# In[9]:


dataset.value_counts()


# In[10]:


dataset.groupby('status').mean()


# In[11]:


dataset.groupby('status').median()


# In[12]:


dataset.groupby('status').agg(['mean','median'])


# Pre-Processing the data

# In[13]:


x=dataset.drop(columns=['name','status'],axis=1)
y=dataset['status']


# In[14]:


y


# In[15]:


x.head()


# In[16]:


x.info()


# In[17]:


scaler=StandardScaler()


# In[18]:


scaler.fit(x)


# In[19]:


x=scaler.transform(x)


# In[20]:


x


# In[21]:


#splitting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)


# # Model Fitting

# ## SVC

# In[22]:


from sklearn import svm


# In[23]:


model=svm.SVC(kernel='linear')


# In[24]:


model.fit(x_train,y_train)


# In[26]:


#accuracy of model on training data
x_train_result=model.predict(x_train)


# In[27]:


train_accuracy=accuracy_score(y_train,x_train_result)


# In[28]:


train_accuracy


# In[29]:


model.fit(x_test,y_test)


# In[30]:


x_test_result=model.predict(x_test)


# In[31]:


test_accuracy=accuracy_score(y_test,x_test_result)


# In[32]:


test_accuracy


# In[33]:


#Using confusion matrix
from sklearn import metrics


# In[34]:


print(metrics.confusion_matrix(y_test, x_test_result))


# In[35]:


print(metrics.classification_report(y_test,x_test_result))


# ## KNN Classifier

# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)


# In[38]:


x_train_result=model.predict(x_train)


# In[39]:


train_accuracy=accuracy_score(y_train,x_train_result)


# In[40]:


train_accuracy


# In[41]:


model.fit(x_test,y_test)


# In[42]:


x_test_result=model.predict(x_test)


# In[43]:


test_accuracy=accuracy_score(y_test,x_test_result)


# In[44]:


test_accuracy


# In[45]:


print(metrics.confusion_matrix(y_test, x_test_result))


# In[46]:


print(metrics.classification_report(y_test,x_test_result))


# ## Decision Tree Classifier

# In[47]:


from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[48]:


model=DecisionTreeClassifier(max_depth = 3, random_state = 1)


# In[49]:


model.fit(x_train,y_train)


# In[50]:


x_train_result=model.predict(x_train)


# In[51]:


train_accuracy=accuracy_score(y_train,x_train_result)


# In[52]:


train_accuracy


# In[53]:


model.fit(x_test,y_test)


# In[54]:


x_test_result=model.predict(x_test)


# In[55]:


test_accuracy=accuracy_score(y_test,x_test_result)


# In[56]:


test_accuracy


# In[57]:


print(metrics.confusion_matrix(y_test,x_test_result))


# In[58]:


print(metrics.classification_report(y_test,x_test_result))


# ## Gaussian NB Classifier

# In[59]:


from sklearn.naive_bayes import GaussianNB


# In[60]:


model=GaussianNB()


# In[61]:


model.fit(x_train,y_train)


# In[62]:


x_train_result=model.predict(x_train)


# In[63]:


train_accuracy=accuracy_score(y_train,x_train_result)


# In[64]:


train_accuracy


# In[65]:


model.fit(x_test,y_test)


# In[66]:


x_test_result=model.predict(x_test)


# In[67]:


test_accuracy=accuracy_score(y_test,x_test_result)


# In[68]:


test_accuracy


# In[69]:


print(metrics.confusion_matrix(y_test,x_test_result))


# In[70]:


print(metrics.classification_report(y_test,x_test_result))


# ## Logistic Regression

# In[71]:


from sklearn.linear_model import LogisticRegression


# In[72]:


model=LogisticRegression()


# In[73]:


model.fit(x_train,y_train)


# In[74]:


x_train_result=model.predict(x_train)


# In[75]:


train_accuracy=accuracy_score(y_train,x_train_result)


# In[76]:


train_accuracy


# In[77]:


model.fit(x_test,y_test)


# In[78]:


x_test_result=model.predict(x_test)


# In[79]:


test_accuracy=accuracy_score(y_test,x_test_result)


# In[80]:


test_accuracy


# In[81]:


print(metrics.confusion_matrix(y_test,x_test_result))


# In[82]:


print(metrics.classification_report(y_test,x_test_result))


# Hence, Decision Tree Classifier is best suited.

# In[83]:


from sklearn import tree


# In[84]:


model=DecisionTreeClassifier(max_depth = 3, random_state = 1)


# In[85]:


model.fit(x_train,y_train)


# In[86]:


model.fit(x_test,y_test)


# In[87]:


tree.plot_tree(model,filled=True)


# # Predicitng Model

# In[88]:


data=input()
data=tuple(float(x) for x in data.split(","))


# In[89]:


#making data into a numpy array
data=np.asarray(data)


# In[90]:


#reshaping the data
data=data.reshape(1,-1)


# In[ ]:


#Standardizing the data
#data=scaler.fit(data)


# In[91]:


data=scaler.transform(data)


# In[92]:


prediction=model.predict(data)


# In[94]:


prediction


# In[95]:


if(prediction[0]==0):
    print("Patient is healthy")
else:
    print("Patient has Parkinson's Disease")

