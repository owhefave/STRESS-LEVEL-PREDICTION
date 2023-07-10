
#!/usr/bin/env python
# coding: utf-8

#                                            IMPORTING THE REQUIRED LIBRARY

# In[41]:


import pandas as pd
import numpy as np


# In[42]:




#                                     READING THE DATASET AND GETTING TO KNOW THE DATASET

# In[43]:


df= pd.read_csv("Stress-Lysis.csv")


# In[44]:


df.head(5)


# In[45]:


df.shape


# In[46]:


df["Stress Level"].nunique()


# In[47]:


"""
    0 stands for low stress
    1 stands for medium stress
    2 stands for high stress
"""
df["Stress Level"].unique()


# In[48]:


df.info()


# In[49]:


#Statistical Analysis of each columns
df.describe()


# In[50]:


"""
        From the correlation table below
        the target column which is the Stress Level is highly positively correlated with other factors
        This simply means that as this factors increases, there is a corresponding increase in the stress level
"""
df.corr()


#                                                    PREPROCESSING
#                 Scaling the dataset so their distance apart are close so as not have unsual impact on the model

# In[51]:


from sklearn.preprocessing import StandardScaler


# In[52]:


scalar= StandardScaler()


# In[53]:


scalar.fit(df.drop("Stress Level",axis=1))


# In[54]:


scaled_feature= scalar.transform(df.drop("Stress Level",axis=1))


# In[55]:


scaled_feature


# In[56]:


df.columns


# In[57]:


df_new= pd.DataFrame(scaled_feature, columns=['Humidity', 'Temperature', 'Step count'])


# In[58]:


df_new.head(5)


#                                                BUILDING A MODEL

# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X= df_new
y= df["Stress Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[61]:


from sklearn.linear_model import LogisticRegression


# In[62]:


model= LogisticRegression()


# In[63]:


model.fit(X_train, y_train)


# In[64]:


prediction= model.predict(X_test)


# In[65]:


from sklearn.metrics import classification_report


# In[66]:


print(classification_report(y_test,prediction))


# In[67]:


from sklearn.metrics import confusion_matrix


# In[68]:


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

# In[ ]:




