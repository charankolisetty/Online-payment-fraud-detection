#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report


# In[ ]:


data = pd.read_csv(r"C:\Users\Charan\Desktop\jupyter py\online fraud detection.csv")
data


# In[ ]:


data['isFraud'].value_counts()


# In[ ]:


# # converting imbalanced data to balanced
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(random_state=0)
# x_ros , y_ros = ros.fit_sample(X,Y)


# In[ ]:


data.isnull()


# In[ ]:


data.Type.value_counts()


# In[ ]:


data['Type'] = data['Type'].map({"PAYMENT":0,"TRANSFER":1,"CASH_IN":2,"CASH_OUT":3,"DEBIT":4})
data['isFraud'] = data['isFraud'].map({0:"no fraud",1:"fraud"})


# In[ ]:


X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values


# In[ ]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X ,Y ,test_size = 0.3 ,random_state = 0)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(x_train,y_train)


# In[ ]:


y_pred1 = rfc.predict(x_test)
y_pred1


# In[ ]:


print(accuracy_score(y_test,y_pred1)*100)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt = dt.fit(x_train,y_train)


# In[ ]:


y_pred2 = dt.predict(x_test)
y_pred2


# In[ ]:


df = pd.DataFrame(y_pred2)
df
df[:,:1].to_csv('online fraud detection.csv')


# In[ ]:


print(accuracy_score(y_test,y_pred2)*100)


# In[ ]:


cm = confusion_matrix(y_test,y_pred2)
sns.heatmap(cm , annot=True , cmap='Blues')
print(classification_report(y_test,y_pred2))

