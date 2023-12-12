#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np


# In[11]:


import pandas as pd


# In[12]:


dataset = pd.read_csv("Social_Network_Ads.csv")


# Basic Data Analysis

# In[17]:


dataset


# In[14]:


dataset.head()


# In[18]:


dataset.tail()


# In[19]:


dataset.shape


# In[20]:


dataset.describe()


# Pre Process Data

# In[21]:


dataset["Gender"].replace(["Female","Male"],[0, 1], inplace = True)


# In[24]:


dataset.head()


# In[25]:


dataset.tail()


# Import Machine Learning Libraries

# In[35]:


#Preparing our Dataset
X = dataset.iloc[:, 1:4]
y = dataset.iloc[:, 4] 


# In[30]:


#Splitting the dataset into a training and testing dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1000)


# In[36]:


#Normalisation
from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)


# In[37]:


# Comparing how the different classification algorithms will perform
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

logistic_classifier = LogisticRegression()
decision_classifier = DecisionTreeClassifier()
svm_classifier = SVC()
knn_classifier = KNeighborsClassifier()
naive_classifier = GaussianNB()

# Using these classifiers to fit our data, X_train and y_train
logistic_classifier.fit(X_train, y_train)
decision_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
naive_classifier.fit(X_train, y_train)


# In[38]:


# Predicting the test set results
logistic_y_prediction = logistic_classifier.predict(X_test)
decision_y_prediction = decision_classifier.predict(X_test)
svm_y_prediction = svm_classifier.predict(X_test)
knn_y_prediction = knn_classifier.predict(X_test)
naive_y_prediction = naive_classifier.predict(X_test)


# In[39]:


# Printing the evaluation metrics to determine the accuracy of classifiers
from sklearn.metrics import classification_report, accuracy_score

print(accuracy_score(logistic_y_prediction, y_test))
print(accuracy_score(decision_y_prediction, y_test))
print(accuracy_score(svm_y_prediction, y_test))
print(accuracy_score(knn_y_prediction, y_test))
print(accuracy_score(naive_y_prediction, y_test))


# In[40]:


# Printing the classification report
print('Logistic classifier:')
print(classification_report(y_test, logistic_y_prediction))

print('Decision Tree classifier:')
print(classification_report(y_test, decision_y_prediction))

print('SVM Classifier:')
print(classification_report(y_test, svm_y_prediction))

print('KNN Classifier:')
print(classification_report(y_test, knn_y_prediction))

print('Naive Bayes Classifier:')
print(classification_report(y_test, naive_y_prediction))


# In[41]:


# Using a confusion matrix to determine the accuracy of our model
from sklearn.metrics import confusion_matrix

print('Logistic Regression classifier:')
print(confusion_matrix(logistic_y_prediction, y_test))

print('Decision Tree classifier:')
print(confusion_matrix(decision_y_prediction, y_test))

print('KNN Classifier:')
print(confusion_matrix(knn_y_prediction, y_test))

print('SVM classifier:')
print(confusion_matrix(svm_y_prediction, y_test))

print('Naive Bayes classifier:')
print(confusion_matrix(naive_y_prediction, y_test))


# In[42]:


# Making a new prediction & comparing results
new_case = [[0, 60, 2500]] # Gender, Age, Salary

# We will need to transform our new case
new_case = norm.transform(new_case)

print('Logistic Regression classifier', logistic_classifier.predict(new_case))
print('Decision Tree classifier:', decision_classifier.predict(new_case))
print('SVM classifier:', svm_classifier.predict(new_case))
print('KNN classifier:', knn_classifier.predict(new_case))
print('Naive Bayes classifier:', naive_classifier.predict(new_case))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




