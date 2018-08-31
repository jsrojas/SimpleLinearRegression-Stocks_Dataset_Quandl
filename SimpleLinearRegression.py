'''
This script implements a linear regresion and a SVR algorithm applied to a stocks dataset downloaded
from Quandl. In the example the feature Adj. Close is predicted 10 days into the future.
'''
import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

#This defines the style to use on a plot with matplotlib
style.use('ggplot')

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
#from the dataset will predict the avg close price for 342.4 days into the future, 0.00876168224 for 30 days in the future
# 0.10660046729 for 365 days into the future, 0.00292056074 for 10 days into the future
forecast_out = int(math.ceil(0.00292056074*len(df)))

#Here the label column is created, assigned to be equal to the avg close price
#and the rows shifted upwards(negative) the number of days we want to predict
#in the future
df['Label'] = df[forecast_column].shift(-forecast_out)



#Now we create a numpy array and store the data removing the label column
#in order to use them as the training data of the algorithm
#The axis = 1 indicates the drop() function to remove a column, for a row it will be axis=0
X = np.array(df.drop(['Label'], axis=1))

#With the scale() function we normalize the training data between -1 and 1
#this helps to save up resources and time in the training stage
X = preprocessing.scale(X)

#Now we extract the the rows of the last 10 days from the dataset so it becomes the unknown data to predict
X_lately = X[-forecast_out:]

#Print the future 10 days instances without the label and normalized
print(X_lately)

#And now we create the training set X without the data from the 10 last days
X = X[:-forecast_out:]

#This removes the last rows that have NaN spaces in the label column due to the shift
df.dropna(inplace=True)

#Later in another numpy array we store the labels that are going to be predicted
#In this case since it is regression it is the prediction of the average close
#price of a stock 10 days into the future.
y = np.array(df['Label'])



#It is really important to be sure that the number of instances in the labels
#and the training set are the same.
print('Number of instances in the training set: ', len(X),'\nNumber of instances in the test set: ', len(y))


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

#print('Simple Linear Regression Accuracy for', forecast_out,'days into the future is: ',accuracy)
#print('Support Vector Regression Accuracy for', forecast_out,'days into the future is: ',accuracy_SVR)

#We can predict the price using the unknown data with the predict function and the simple linear regression algorithm
#the forecast_set will store the predicted prices of the next 10 days as a list
forecast_set = classifier.predict(X_lately)

#Here we print the predicted prices, the precision obtained by the simple linear regression algorithm and the number
#of days predicted into the future.
print(forecast_set, accuracy, forecast_out)

#Now if we want to graph the training data and the predicted data using matplotlib
#we create a new columnn named Forecast and fill it with nan
df['Forecast'] = np.nan

#Since we are predicting the price of the future 10 days, those instances are not on the original dataframe
#hence we must create those new instances and fill each column
#To do this first we store in last_date the last date found on the dataframe which is 2018-03-13
last_date = df.iloc[-1].name
print(df.tail())
print('last_date', last_date)

#Now we convert that last date into unix time in order to process it as an integer
last_unix = last_date.timestamp()
print('last_unix: ', last_unix)

#And now we move to the next day adding the total of seconds of a day (86400 seconds) to the last date
#therefore next_unix will be the date 2018-03-14
one_day = 86400
next_unix = last_unix + one_day

#Now using a for loop we extract the predictions of the next 10 days stored in forecast_set
#and we add it to the new instance created for each new date
for i in forecast_set:
    #This line converts from unix time to date format
    next_date = datetime.datetime.fromtimestamp(next_unix)
    #This line moves to the next day
    next_unix += one_day
    #This line creates the new instance in the dataframe. 
    #The loc[] function puts the current next date as the value of the first column
    #and then all the other columns are filled with nan values except the forecast column (last column)
    #the forecast column is added using the i value found on the forecast set
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+ [i]

#this print shows the new instances created with nan values and the prediction of the next ten days starting on 2018-03-14
print(df.tail())

#Finally using matplotlib we plot the Adj. Close column (training data) and the Forecast column (predicted data for the next 10 days)
df['Adj. Close'].plot()
df['Forecast'].plot()
#this line puts the graph labels on the bottom right of the plot
plt.legend(loc=4)
#this lines puts the labels to both axis of the plot
plt.xlabel('Date')
plt.ylabel('Price')
#Finally this line shows the plot
plt.show()
