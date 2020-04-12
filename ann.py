# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pdb;pdb.set_trace()
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# X values start at 3 because we don't care about row number, surname, etc.
X = dataset.iloc[:, 3:13].values
# y value is the final column (Exited {0,1})
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
#Label encode the second column of X (Geography)
#Converts country strings into integers (i.e. France = 0, Spain = 1, Germany = 2)
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
#Label encode the third column of X (Gender)
#Converts gender strings into integers (i.e. Male = 0, Female = 1)
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#One hot encode the geography where each category has its own column

#Create a dummy variable at column 1
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Remove the second column from X to prevent the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'RandomNormal', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'RandomNormal', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'RandomNormal', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 50, epochs = 100)

# Part 3 - Making predictions and evaluating the model


# Converts array into True or False based on the threshold value
ideal_thresh = 0
max_accuracy = 0
for i in range (1,101):

	# Predicting the Test set results
	y_pred = classifier.predict(X_test)

	threshold = i / 100
	y_pred = (y_pred > threshold)

	# Making the Confusion Matrix
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y_test, y_pred)
	acc = (cm[0][0] + cm[1][1]) / 2000
	print("Accuracy: "),
	print(acc)

	if acc > max_accuracy:
		ideal_thresh = threshold
		max_accuracy = acc

print(max_accuracy, ideal_thresh)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0.0, 600.0, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print(new_prediction > ideal_thresh)
