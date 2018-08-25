'''
This script implements a linear regresion and a SVR algorithm applied to a stocks dataset downloaded
from Quandl. In the example the feature Adj. Close is predicted 10 days into the future.
'''
import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#The dataset holding stock information from google is downloaded
#using the quandl module and the get() function
df = quandl.get('WIKI/GOOGL')

#Now after seeing the features or attributes that are more interesting
#we reduce the dataframe to those attributes only
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PERCENT'] = (df['Adj. High']- df['Adj. Low'])/df['Adj. Low']*100.0
df['PCT_change'] = (df['Adj. Close']- df['Adj. Open'])/df['Adj. Open']*100.0

df = df[['Adj. Close', 'HL_PERCENT', 'PCT_change', 'Adj. Volume']]

forecast_column = 'Adj. Close'

#This replaces the NaN values, if any, with -9999
#The inplace parameter, if set to true allows to change the object
#that holds the dataset
df.fillna(-9999, inplace=True)

#This variable defines how many days into the future the average close price
#will be predicted by taking a percentage of instances from the dataset.
#For example, since the dataset has 3424 instances taking 10% (0.1) instances
#from the dataset will predict the avg close price for 342 days into the future
forecast_out = int(math.ceil(0.00292056074*len(df)))

#Here the label column is created, assigned to be equal to the avg close price
#and the rows shifted upwards(negative) the number of days we want to predict
#in the future
df['Label'] = df[forecast_column].shift(-forecast_out)

#This removes the last rows that have NaN spaces in the label column due to the shift
df.dropna(inplace=True)

#Now we create a numpy array and store the data removing the label column
#in order to use them as the training data of the algorithm
#The axis = 1 indicates the drop() function to remove a column, for a row it will be axis=0
X = np.array(df.drop(['Label'], axis=1))

#Later in another numpy array we store the labels that are going to be predicted
#In this case since it is regression it is the prediction of the average close
#price of a stock 10 days into the future.
y = np.array(df['Label'])

#With the scale() function we normalize the training data between -1 and 1
#this helps to save up resources and time in the training stage
X = preprocessing.scale(X)

#It is really important to be sure that the number of instances in the labels
#and the training set are the same.
print(len(X), len(y))


#Now we define the train and test sets and with cross validation pass the data
#and define a 20% of the data for the testing stage.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#And now we define our classifier as linear regression and fit the data
#to perform the training process of the classifier.
classifier = LinearRegression()
classifier.fit(X_train, y_train)

#We can test another algorithm like support vector regression like this
classifier_SVR = svm.SVR()
classifier_SVR.fit(X_train, y_train)

#Now in order to test the classifiera we use the score() function
#like this
accuracy = classifier.score(X_test, y_test)
accuracy_SVR = classifier_SVR.score(X_test, y_test)

print('Simple Linear Regression Accuracy for', forecast_out,'days into the future is: ',accuracy)
print('Support Vector Regression Accuracy for', forecast_out,'days into the future is: ',accuracy_SVR)
