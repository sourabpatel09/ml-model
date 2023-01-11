#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,accuracy_score


# In[2]:


fruits = pd.read_excel('c:/data/fruits.xlsx',sheet_name='fruits')
fruits.shape


# In[3]:


X = fruits[['Sphericity','Weight']]
y = fruits['labels']


# In[98]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=4)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[108]:


plt.scatter(x='Sphericity',y='Weight',data=X_train[y_train=='Apple'],c='red',label='Train Apple')
plt.scatter(x='Sphericity',y='Weight',data=X_train[y_train=='Orange'],c='orange',label='Train Orange')
plt.scatter(x='Sphericity',y='Weight',data=X_test,c='blue',marker='*',label='Test Samples')
plt.plot([0.7,0.95],[150.5,150.5],c='black')
plt.plot([0.796,0.796],[100,150.5],c='black')
plt.legend()
plt.show()


# In[109]:


y_train.value_counts(),y_test.value_counts()


# In[111]:


clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[107]:


plt.figure(figsize=[8,10])
plot_tree(clf,feature_names=['Sphericity','Weight'],class_names=['Apple','Orange'])
plt.show()


# In[112]:


y_predict = clf.predict(X_test)
accuracy_score(y_test,y_predict)


# In[104]:


confusion_matrix(y_test,y_predict)


# In[105]:


plot_confusion_matrix(clf,X_test,y_test,colorbar=False)
plt.show()


# In[106]:


plt.scatter(x='Sphericity',y='Weight',data=X_train[y_train=='Apple'],c='red',label='Train Apple')
plt.scatter(x='Sphericity',y='Weight',data=X_train[y_train=='Orange'],c='orange',label='Train Orange')
plt.scatter(x='Sphericity',y='Weight',data=X_test[y_test==y_predict],c='green',marker='*',s=100,label='correctly classified')
plt.scatter(x='Sphericity',y='Weight',data=X_test[y_test!=y_predict],c='blue',marker='*',s=100,label='Wrongly classified')
#plt.plot([0.7,0.95],[150.5,150.5],c='black')
#plt.plot([0.78,0.78],[100,150.5],c='black')
plt.legend()
plt.show()

