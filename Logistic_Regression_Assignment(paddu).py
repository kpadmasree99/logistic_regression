#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression

# this is a Classification model even though it has a regression in its name it is mainly used for classifaction with the help of sigmoid algorithm 
# sigmoid = 1/1+e-z
# z stands for distance * value 

# In[1]:


#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,confusion_matrix,classification_report


# ### Train Data

# In[2]:


#the datasets we got from the team has separate files for training and testing 
traindata1 = pd.read_csv('Titanic_train.csv')
traindata = traindata1.copy()
traindata.head()


# In[3]:


traindata.shape


# In[4]:


traindata.columns


# In[5]:


traindata.hist(bins=10,figsize=(9,7),grid=False)
plt.show()


# In[6]:


sns.set(font_scale=1)
sns.countplot(x='Sex',data=traindata,palette='plasma').set_title('how many men and women in traindata')


# As the data is having few categorical columns and they are not related to our analysis (PassengerID,name,Ticket) we remove them and for other columns we gonna have dummies and check
# 

# ### EDA for Train data

# In[7]:


#dropping the columns PassengerId,Name,Ticket as they are categorical and doesnt correlate with the target column
traindata.drop(columns=['Name','Ticket'],inplace=True)


# In[8]:


traindata


# In[9]:


#getting dummies for categorical columns in the dataframe
dummiestrain= pd.get_dummies(traindata[['Sex','Embarked']],dtype=int)
dummiestrain


# In[10]:


traindata.drop('Sex',axis=1,inplace=True)


# In[11]:


traindata.drop('Embarked',axis=1,inplace=True)


# In[12]:


traindata.drop('Cabin',axis=1,inplace=True)


# In[13]:


traindata=dummiestrain.join(traindata,how='right')


# In[14]:


traindata


# In[15]:


traindata.isnull().sum()


# In[16]:


traindata['Age']=traindata['Age'].fillna(traindata['Age'].median())


# In[17]:


traindata.isnull().sum()


# In[18]:


traindata[['Age','Fare']]=np.int64(traindata[['Age','Fare']])


# In[19]:


traindata.info()


# In[20]:


#correlation 
sns.heatmap(traindata.corr(),annot=True,cmap='coolwarm',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(25,15)
plt.show()


# In[21]:


#check for duplicate columns


# In[22]:


traindata.duplicated().sum()


# In[23]:


features_train = traindata.drop('Survived',axis=1)
features_train


# In[24]:


target_train= traindata[['Survived']]
target_train


# ### Test Data

# In[25]:


testdata = pd.read_csv('Titanic_test.csv')


# In[26]:


testdata


# In[27]:


testdata.hist(bins=10,grid=False,figsize=(9,7))
plt.show()


# ### perform EDA for the test data

# In[28]:


testdata.drop(columns=['Name','Ticket','Cabin'],inplace=True)


# In[29]:


testdata


# In[30]:


dummies = pd.get_dummies(testdata[['Sex','Embarked']],dtype=int)


# In[31]:


testdata.drop('Sex',axis=1,inplace=True)


# In[32]:


testdata.drop('Embarked',axis=1,inplace=True)


# In[33]:


testdata=dummies.join(testdata,how='right')


# In[34]:


testdata['Age'].dtype


# In[35]:


testdata.info()


# In[36]:


testdata[['Age','Fare']] = np.int64(testdata[['Age','Fare']])


# In[37]:


testdata.head()


# ## Train The Model Logistic Regression

# In[38]:


logreg = LogisticRegression()


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(features_train,target_train,train_size=0.8,random_state=10)


# In[40]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[41]:


logreg.fit(x_train,y_train)


# In[42]:


ypred= logreg.predict(x_test)
ypred


# In[43]:


print(logreg.score(x_test,y_test))
ypred = logreg.predict(x_test)
print("suvived",sum(ypred!=0))
print("Not Survived",sum(ypred==0))


# In[44]:


#confusion matrix


# In[45]:


confusion_matrix(y_test,ypred)


# In[46]:


sns.heatmap(confusion_matrix(y_test,ypred),annot=True,fmt='3.0f')


# ### Accuracy

# In[47]:


score=round(accuracy_score(ypred,y_test)*100,2)
print(f"The accuracy for the model is {score}")


# ## Classification Report

# In[48]:


classification_report(y_test,ypred)


# In[55]:


roc_auc = roc_auc_score(y_test,ypred)
print(roc_auc)


# In[61]:


fpr,tpr,thr = roc_curve(y_test,ypred)
plt.plot(fpr,tpr,lw=2,color='blue',label=f'AUC:{roc_auc:.2}')
plt.plot([0,1],[0,1],color='red',linestyle='--')
plt.xlabel('fpr-false_positive_rate')
plt.ylabel('tpr-True_positive_rate')
plt.title('ROC_Curve')
plt.grid(False)


# In[ ]:




