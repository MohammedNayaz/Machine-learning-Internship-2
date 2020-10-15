#!/usr/bin/env python
# coding: utf-8

# # Project: Youtube adview Prediction

# # 1.Importing the datasets and libraries, check shape and datatype.

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#importing the datasets
data_train = pd.read_csv("train.csv")


# In[3]:


#Displaying the imported dataset
data_train


# In[4]:


#checking the shape of dataset
data_train.shape


# In[5]:


#checking the head part of the dataset
data_train.head()


# In[6]:


#checking the tail part of the dataset
data_train.tail()


# In[7]:


#describing the dataset
data_train.describe


# In[8]:


#checking the datatype of the dataset
type(data_train)


# # 2.Visualizing the dataset using plotting, using heatmaps abd plots

# In[9]:


#visualization of the dataset

#2D plotting
  #Individual plots

#line graph 
plt.plot(data_train["adview"])
plt.show()


# In[10]:


plt.plot(data_train["views"])
plt.show()


# In[11]:


plt.plot(data_train["likes"])
plt.show()


# In[12]:


plt.plot(data_train["dislikes"])
plt.show()


# In[13]:


plt.plot(data_train["comment"])
plt.show()


# In[14]:


plt.plot(data_train["category"])
plt.show()


# In[15]:


plt.plot(data_train["duration"])
plt.show()


# In[16]:


#plotting histogram plot
plt.hist(data_train["category"])
plt.show()


# In[17]:


#ploting the 2D graph before removal of the outlier
plt.plot(data_train["adview"])
plt.show()


# In[18]:


#Considering the adview column as per the objective of this model
#removing the outliers: lets remove thevideos with adview grater than 2000000
data_train = data_train[data_train["adview"]<2000000]


# In[19]:


#ploting the 2D graph to check the removal of the outlier
plt.plot(data_train["adview"])
plt.show()


# In[20]:


#Ploting Heatmap
f, ax = plt.subplots(figsize=(10,8))
corr = data_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True)
plt.show()


# # 3.Cleaning the datasets by removing missing values and other things

# In[21]:


#removing chharacter "F" present in data
data_train = data_train[data_train.views != 'F']
data_train = data_train[data_train.likes != 'F']
data_train = data_train[data_train.dislikes != 'F']
data_train = data_train[data_train.comment != 'F']

data_train.head()


# In[22]:


data_train.tail(20)


# In[23]:


#Assigning each category a number for category feature
category = {'A':1 , 'B':2 , 'C':3 , 'D':4 , 'E':5 , 'F':6 , 'G':7 , 'H':8}
data_train["category"] = data_train["category"].map(category)

#Dataset after cleaning and categorizing the features
data_train.head()


# In[24]:


data_train.tail(20)


# # 4.Transforming the attributes into numerical values and other necessary transformations

# In[25]:


#Converting all floating point numbers to integers for views, likes, comments, dislikes, and adviews

data_train["views"] = pd.to_numeric(data_train["views"])
data_train["comment"] = pd.to_numeric(data_train["comment"])
data_train["likes"] = pd.to_numeric(data_train["likes"])
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"])
data_train["adview"] = pd.to_numeric(data_train["adview"])

column_vidid = data_train["vidid"]


# In[26]:


#Encoding features like category, duration and vidid

from sklearn.preprocessing import LabelEncoder
data_train['duration'] = LabelEncoder().fit_transform(data_train['duration'])
data_train['vidid'] = LabelEncoder().fit_transform(data_train['vidid'])
data_train['published'] = LabelEncoder().fit_transform(data_train['published'])

data_train.head()


# In[27]:


data_train.tail(10)


# In[28]:


#Converting Time_in_sec for duration
import datetime
import time

#defining a functionc"checki" to check the duration of the adds
def checki(x):
  y = x[2:]
  h = ''
  m = ''
  s = ''
  mm = ''
  p = ['H' , 'M' , 'S']
  
  for i in y:
    if i not in p:
      mm+=i
    else:
      if (i == "H"):
        h = mm
        mm = ''
      elif (i == "M"):
        m = mm
        mm = ''
      else:
        s = mm
        mm = ''
  
  if (h == ''):
    h = '00'
  if (m == ''):
    m = '00'
  if (s == ''):
    s = '00'

  bp = h+':'+m+':'+s
  return bp

#reloading the csv file to the project and assiging it to train
train = pd.read_csv("train.csv")
mp = pd.read_csv("train.csv")["duration"]

#applying the "checki" function to the csv file
time = mp.apply(checki)

#defining a function "func_sec" to transform the time into seconds
def func_sec(time_string):
  h, m, s =time_string.split(':')
  return int(h) * 3600 + int(m) * 60 + int(s)

#calling the function "func_sec" and applying to the above function variable _time_
time1 = time.apply(func_sec)

#applying the above all functions to the the dataset ("data_train")
data_train["duration"] = time1

#displaying the newly modified dataset
data_train.head()


# In[29]:


data_train.tail(10)


# In[30]:


#once again plotting the graph and checking the dataset
#ploting the 2D graph to check the removal of the outlier
plt.plot(data_train["adview"])
plt.show()


# In[31]:


#Ploting Heatmap after succefully cleaning the dataset and transforming the necessary attributes
f, ax = plt.subplots(figsize=(10,8))
corr = data_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True)
plt.show()


# # 5.Normalizing the data and spliting the data into training, validation and testset in the appropriate ratio

# In[32]:


#Spliting the data into 80:20 ratio

Y_train = pd.DataFrame(data = data_train.iloc[: , 1].values , columns = ['target'])
data_train = data_train.drop(["adview"] , axis = 1)
data_train = data_train.drop(["vidid"] , axis = 1)
data_train.head()


# In[33]:


Y_train


# In[34]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split ( data_train , Y_train , test_size = 0.2 , random_state = 42)


# In[35]:


x_train


# In[36]:


x_train.shape


# In[37]:


x_test


# In[38]:


x_test.shape


# In[39]:


y_train 


# In[40]:


y_train.shape


# In[41]:


y_test


# In[42]:


y_test.shape


# In[43]:


#Split of 80:20 ratio is :.... considering the 80% of the data set (14999 - 14636)
11708 + 2928


# In[44]:


#Normaliseing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# In[45]:


x_train.mean()


# In[46]:


x_train , y_train


# In[47]:


x_test , y_test


# In[48]:


print(train.columns)


# In[49]:


EDA_train = pd.read_csv('train.csv')


# In[51]:


EDA_train['adview'].value_counts()


# In[52]:


EDA_train.plot(kind="scatter",x="adview",y="likes")
plt.show()


# # 6. Using Linear regression, Support vector Regressor for training and get errors

# In[53]:


#Evaluation metrics
from sklearn import  metrics

def print_error(x_test, y_test, model_name):
    prediction = model_name.predict(x_test)
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[54]:


#Linear Regression

from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(x_train, y_train)
print_error(x_test, y_test, linear_regression)


# In[55]:


data_train.describe()


# In[56]:


data_train.isnull().any()


# In[57]:


linear_regression = linear_model.LinearRegression()
linear_regression.fit(x_train, y_train)
print_error(x_test, y_test, linear_regression)


# In[58]:


#Support vector Regressioin
from sklearn.svm import SVR
supportvector_regressor = SVR()
supportvector_regressor.fit(x_train, y_train)
print_error(x_test , y_test , supportvector_regressor)


# # 7.Using Decision Tree Regressor and Random Forest Regressors.

# In[59]:


#Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(x_train ,y_train)
print_error(x_test , y_test , decision_tree)


# In[60]:


#Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
n_estimators = 200
max_depth = 25
min_samples_split = 15
min_samples_leaf = 2
random_forest = RandomForestRegressor(n_estimators = n_estimators , max_depth = max_depth, min_samples_split =  min_samples_split  , min_samples_leaf = min_samples_leaf )
random_forest.fit(x_train,y_train)
print_error(x_test , y_test , random_forest)


# # 8.Building An Artificial Neural Network & Training it With Different Layers & Hyperparameteres using Keras

# In[61]:


#Artiificial Neural Network

import keras
from keras.layers import Dense

ann = keras.models.Sequential([
                              Dense(6,activation = "relu", 
                              input_shape = x_train.shape[1:]),
                              Dense(6, activation = "relu"), 
                              Dense(1)
                             ])

optimizer = keras.optimizers.Adam()
loss = keras.losses.mean_squared_error
ann.compile(optimizer = optimizer , loss = loss , metrics = ["mean_squared_error"])

history = ann.fit(x_train , y_train , epochs = 100)


# In[62]:


ann.summary()


# In[63]:


print_error(x_test , y_test , ann)


# # 9.Picking the best Model based on error as well as generalisation.

# In[64]:


# 1. Lineae Regression
print_error(x_test, y_test, linear_regression)


# In[65]:


# 2. Support Vector Regressor
print_error(x_test , y_test , supportvector_regressor)


# In[66]:


# 3. Decision Tree Regressor
print_error(x_test , y_test , decision_tree)


# In[67]:


# 4. Random Forest Regressor
print_error(x_test , y_test , random_forest)


# #Decision tree regression is best Model based on errors and also by gerneralization.

# # 10.Saving the model and predicting on test set.

# In[69]:


#Saving Scikitlearn models
import joblib
joblib.dump(decision_tree , "decisiontree_youtubeadview.pkl")


# In[70]:


#Saving keras Artificial Neural Network Model
ann.save("ann_youtubeadview.h5")

