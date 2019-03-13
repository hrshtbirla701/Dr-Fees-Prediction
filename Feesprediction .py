# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#importing all packages
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


#importing data
file = pd.ExcelFile('Final_Train.xlsx')
file.sheet_names
dataset = file.parse('Sheet1')


#dividing Place coulm into Place and city
new = dataset["Place"].str.split(",", n = 1, expand = True)
dataset["Area"]= new[0] 
dataset["City"]= new[1]
dataset.drop(['Place'],axis = 1,inplace = True)
dataset['Area'] = [dataset[dataset['Area'] == x]['Fees'].mean() for x in dataset['Area']]


cols = list(dataset.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Fees')) #Remove b from list
dataset = dataset[cols+['Fees']] #Create new dataframe with columns in the order you want



#dividing data into independent and dependent data
dataset_X = dataset.iloc[:,:7].values
dataset_Y = dataset.iloc[:,7].values


#converting profile column into mathematical form
dataset['Profile'].unique()
test_X = dataset['Profile']
test_X =  pd.DataFrame(test_X)
test_X.iloc[:,0]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
test_X.iloc[:,0] = labelEncoder_X.fit_transform(test_X.iloc[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
test_X = onehotencoder.fit_transform(test_X)
X = test_X.todense()
X = pd.DataFrame(X).astype(int)
X = X.iloc[:,1:]  #avoiding dummy variable trap
dataset_X = pd.DataFrame(dataset_X)
dataset_X = pd.concat([dataset_X,X],axis = 1)
dataset_X.columns = ['Qualification','Experience','Rating','Profile','Misc','Area','City','Prof1','Prof2','Prof3','Prof4','Prof5']
dataset_X.drop(['Profile'],axis = 1,inplace = True)


#extracting years in no from the column
dataset_X['Experience'] = [re.findall(r'^[0-9]+',ex)[0] for ex in dataset_X['Experience']]


#dividing qualificaion coulm into degree and qualification
#new = dataset_X["Qualification"].str.split("-", n = 1, expand = True)
dataset_X["Degree"]= [re.findall(r'[A-Z]+\s?-?\s?[A-Z]+\s*',x) for x in dataset_X['Qualification']] 
#dataset_X["Qualification"]= new[1]


#convertinf=g degree column into no of degree
dataset_X['Degree'] = [len(x) for x in dataset_X['Degree']]





#removing City which doesnot have enough data

arr_list = dataset_X[dataset_X['City'].isnull()].index.tolist()
dataset_X = dataset_X.dropna(subset=['City'])  #droping rows where city is null
dataset_X.reset_index(drop=True, inplace=True) # resetting the index into sequence

test_X = dataset_X['City']
test_X =  pd.DataFrame(test_X)
test_X.iloc[:,0]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
test_X.iloc[:,0] = labelEncoder_X.fit_transform(test_X.iloc[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
test_X = onehotencoder.fit_transform(test_X)
X = test_X.todense()
X = pd.DataFrame(X).astype(int)
X = X.iloc[:,1:]  #avoiding dummy variable trap
dataset_X = pd.DataFrame(dataset_X)
dataset_X = pd.concat([dataset_X,X],axis = 1,ignore_index = False)
dataset_X.columns = ['Qualification','Experience','Rating','Misc','Area','City','Prof1','Prof2','Prof3','Prof4','Prof5','Degree','City1','City2','City3','City4','City5','City6','City7','City8']


#removing column which doesnot have enough data
dataset_X.drop(['City','Misc','Rating','Qualification'],axis = 1,inplace = True)


#removing row from dependent data for whom data is deleted in independent data using index column
dataset_Y = pd.DataFrame(dataset_Y)
dataset_Y.drop(dataset_Y.index[arr_list],inplace = True)


#splitting data into train and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset_X, dataset_Y,test_size = 0.50, random_state = 0)


#this is for feature scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train = sc_x.fit_transform(X_train)
Y_train = sc_y.fit_transform(Y_train)
 

#svm algo
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train,Y_train)


#predicting the test data
predict_Y_test = sc_y.inverse_transform(regressor.predict(sc_x.transform(X_test)))


#evaluating square mean error
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(predict_Y_test,Y_test))
print(rms)


#saving the trained model
filename = 'finalized_model.sav'
pickle.dump(regressor, open(filename, 'wb'))