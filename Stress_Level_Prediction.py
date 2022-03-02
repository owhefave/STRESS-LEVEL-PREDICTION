#!/usr/bin/env python
# coding: utf-8

#                                            IMPORTING THE REQUIRED LIBRARY

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#                                     READING THE DATASET AND GETTING TO KNOW THE DATASET

# In[3]:


df= pd.read_csv("Stress-Lysis.csv")


# In[4]:


df.head(5)


# In[5]:


df.shape


# In[6]:


df["Stress Level"].nunique()


# In[7]:


"""
    0 stands for low stress
    1 stands for medium stress
    2 stands for high stress
"""
df["Stress Level"].unique()


# In[8]:


df.info()


# In[9]:


#Statistical Analysis of each columns
df.describe()


# In[10]:


sns.pairplot(df)


# In[11]:


"""
        From the correlation table below
        the target column which is the Stress Level is highly positively correlated with other factors
        This simply means that as this factors increases, there is a corresponding increase in the stress level
"""
df.corr()


#                                                    PREPROCESSING
#                 Scaling the dataset so their distance apart are close so as not have unsual impact on the model

# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


scalar= StandardScaler()


# In[14]:


scalar.fit(df.drop("Stress Level",axis=1))


# In[15]:


scaled_feature= scalar.transform(df.drop("Stress Level",axis=1))


# In[16]:


scaled_feature


# In[17]:


df.columns


# In[18]:


df_new= pd.DataFrame(scaled_feature, columns=['Humidity', 'Temperature', 'Step count'])


# In[19]:


df_new.head(5)


#                                                BUILDING A MODEL

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X= df_new
y= df["Stress Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


model= LogisticRegression()


# In[24]:


model.fit(X_train, y_train)


# In[25]:


prediction= model.predict(X_test)


# In[26]:


from sklearn.metrics import classification_report


# In[27]:


print(classification_report(y_test,prediction))


# In[28]:


from sklearn.metrics import confusion_matrix


# In[29]:


print(confusion_matrix(y_test,prediction))


#                           Interpretation of the confusion Matrix
#                         Predicted Low      Predicted Medium     Predicted High
#     Actual Low              163                  1                   0
#     Actual Medium            0                  220                  0
#     Actual High              0                   1                  216                   
#     
#     163 Predicted low stress as against Actual 164
#     222 Predicted Medium as against the Actual 220
#     216 Predicted High as against the Actual 217
