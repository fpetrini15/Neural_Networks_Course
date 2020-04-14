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
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
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
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def main():

	while(1):
		print("*========================*")
		print("Select the execution type: \n \t1 -> Simple, \n \t2->k-fold Validation,\n \t3-> Grid Optimization, \n \t4-> Exit")
		selection_type = input()
		if selection_type == "1":
			simpleExecution()
		elif selection_type == "2":
			k_foldValidation()
		elif selection_type == "3":
			gridOptimization()
		elif selection_type == "4":
			break
		else:
			print("Valid Option Not Selected")

def simpleExecution():
	
	# Part 2 - Now let's make the ANN!
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

		if acc > max_accuracy:
			ideal_thresh = threshold
			max_accuracy = acc

	print("Max Accuracy: ", max_accuracy)
	print("Ideal Threshold: ", ideal_thresh)

def testNewPrediction():
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

	## The element is contained within two brackets because otherwise it would represent a column
	## Because this is a new prediction, we're inserting a row 
	new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0.0, 600.0, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
	print(new_prediction > ideal_thresh)
	'''
	## Evaluating and improving the model
	'''


def buildClassifier():
	classifier = Sequential()
	classifier.add(Dense(units = 8, kernel_initializer = 'RandomNormal', activation = 'relu', input_dim = 11))
	classifier.add(Dense(units = 8, kernel_initializer = 'RandomNormal', activation = 'relu'))
	classifier.add(Dense(units = 1, kernel_initializer = 'RandomNormal', activation = 'sigmoid'))
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	return classifier

def k_foldValidation(k = 10):
	classifier = KerasClassifier(build_fn = buildClassifier, batch_size = 50, epochs = 100)
	#cv = 10 for 10-fold cross validation || n_jobs = -1 means it will use all CPUs in parallel to get it done faster
	accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = k, n_jobs = -1) 
	print(accuracies)
	mean = accuracies.mean()
	variance = accuracies.std()

def buildClassifierGrid(optimizer_parameter):
	classifier = Sequential()
	classifier.add(Dense(units = 8, kernel_initializer = 'RandomNormal', activation = 'relu', input_dim = 11))
	classifier.add(Dense(units = 8, kernel_initializer = 'RandomNormal', activation = 'relu'))
	classifier.add(Dense(units = 1, kernel_initializer = 'RandomNormal', activation = 'sigmoid'))
	classifier.compile(optimizer = optimizer_parameter, loss = 'binary_crossentropy', metrics = ['accuracy'])
	return classifier

def gridOptimization():
	classifier = KerasClassifier(build_fn = buildClassifierGrid, batch_size = 50, epochs = 100)
	parameters = {'batch_size': [25,32],
				  'epochs': [100,200],
				  'optimizer_parameter': ['adam', 'rmsprop'] }

	gridSearch = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)

	gridSearch = gridSearch.fit(X_train, y_train)
	best_parameters = gridSearch.best_params_
	best_accuracy = gridSearch.best_score_

	print("Best parameters: ", best_parameters)
	print("Best Score: ", best_accuracy)


if __name__ == "__main__":
    main()






