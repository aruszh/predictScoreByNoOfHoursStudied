#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas, numpy
from matplotlib import pyplot


# In[2]:


url = "http://bit.ly/w-data"

dataOfStudents=pandas.read_csv(url)


# In[3]:


print(dataOfStudents.head())


# In[4]:


dataOfHours=dataOfStudents[['Hours']]


# In[5]:


dataOfScores=dataOfStudents[['Scores']]


# In[6]:


pyplot.scatter(dataOfHours,dataOfScores)


# In[7]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[8]:


hours_train,hours_test,score_train,score_test=train_test_split(dataOfHours,dataOfStudents,test_size=0.3)


# In[9]:


lr=LinearRegression()


# In[10]:


lr.fit(hours_train,score_train)


# In[12]:


score_predict=lr.predict(hours_test)


# In[21]:


print(score_predict)


# In[22]:


from sklearn import metrics  


# In[25]:


print('Mean Absolute Error:', 
      metrics.mean_absolute_error(score_test, score_predict))


# In[26]:


pyplot.scatter(score_test, score_predict)


# In[13]:


loop=True
while(loop):
    x=int(input('Enter No Of Hours studied in hrs'))
    if x<=10:
        loop=False
    else:
        print('Please Study less than 8-10 hrs')
        
hr=[[0]]
hr[0][0]=x
print(lr.predict(hr))

